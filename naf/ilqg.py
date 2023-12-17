import numpy as np

class MBA():
    def __init__(self, x_len, u_len, rxu, horizon):
        self.x_len = x_len
        self.u_len = u_len
        self.xu_len = x_len + u_len
        self.rxu = rxu
        self.horizon = horizon
        
    # takes in replay_buffer.ModelingBuffer
    def flld_ilqg(self, B):
        transitions_ts = self._get_transitions_ts(B)
        llds = self._flld(transitions_ts)
        gs = self._ilqg(transitions_ts, llds)
        
        return llds, gs
             
    def _get_transitions_ts(self, B):
        transitions = B.sample_exps(get_all=True)

        # sort transitions by time for llds and ilqg forward pass
        transitions_ts = {}
        for transition in transitions:
            transitions_t = transitions_ts.get(transition.t, [])
            transitions_t.append(transition)
            transitions_ts[transition.t] = transitions_t

        return transitions_ts

    # fit local linear dynamics
    def _flld(self, transitions_ts):
        # creates a new dict each time rather than updating old dict
        llds = {}

        for t, transitions_t in transitions_ts.items():
            llds[t] = self._flld_t(transitions_t)

        return llds


    def _flld_t(self, transitions_t, reg=1e-6):
        # create X = [x u 1]^T
        n = len(transitions_t)
        xs = np.vstack([trans.state for trans in transitions_t])
        us = np.vstack([trans.action for trans in transitions_t])
        xs_next = np.vstack([trans.next_state for trans in transitions_t])
        xs_next = np.transpose(xs_next)  # convert to columnar [x_next]s

        # use columnar [x u 1] as "regular", and row [x u 1] as transpose
        xusT = np.hstack((xs, us, np.ones((n, 1))))  # xus transpos
        xus = np.transpose(xusT)  # convert to columnar [x u 1]s

        # fit linear regression
        # FX = B
        # => F = BX^T (XX^T)^{-1}
        BXT = np.matmul(xs_next, xusT)
        XXT_inv = np.linalg.inv(np.matmul(xus, xusT) + (reg * np.eye(self.xu_len+1)))
        dynamics = np.matmul(BXT, XXT_inv)
        F_t = dynamics[:, :-1]  # linear F_t
        f_t = dynamics[:, -1].flatten()  # intercept term

        return (F_t, f_t)


    def _ilqg(self, transitions_ts, llds):
        # start from horizon and work backwards
        T = self.horizon - 1
        first_iter = True
        
        # create empty lists which will be filled backwards in _ilqg_backward_step()
        Vs = [None for _ in range(self.horizon)]
        vs = [None for _ in range(self.horizon)]
        Ks = [None for _ in range(self.horizon)]
        ks = [None for _ in range(self.horizon)]
        ctrls = [None for _ in range(self.horizon)]
        for t in range(self.horizon-1, -1, -1):
            lld = llds.get(t, None)
            if lld is not None:
                self._ilqg_backward_step(first_iter, T, t, lld, Vs, vs, Ks, ks)
                first_iter = False

        for t in range(self.horizon):
            ctrls[t] = self._ilqg_forward_step(transitions_ts, t, Ks, ks)

        return ctrls

    def _ilqg_backward_step(self, first_iter, T, t, lld, Vs, vs, Ks, ks, reg=1e-6):
        # one step of backward pass
        # calculates matrices for K_t, k_t and appends them to list
        R_t, r_t = self.rxu
        F_t, f_t = lld

        if first_iter:  # use this instead of t == T because rollout might not go until horizon
            Q_t = R_t
            q_t = r_t
        else:
            # calculate Q_t matrix
            VF_t = np.matmul(Vs[t+1], F_t)
            FVF_t = np.matmul(F_t.transpose(), VF_t)
            Q_t = R_t + FVF_t

            # calculate q_t vector
            Vf_t = np.matmul(Vs[t+1], f_t)
            FVf_t = np.matmul(F_t.transpose(), Vf_t)
            Fv = np.matmul(F_t.transpose(), vs[t+1])
            q_t = r_t + FVf_t + Fv

        # sanity check
        assert Q_t.shape == (self.xu_len, self.xu_len)
        assert q_t.shape == (self.xu_len,)

        Qxx = Q_t[:self.x_len, :self.x_len]
        Qxu = Q_t[:self.x_len, self.x_len:]
        Qux = Q_t[self.x_len:, :self.x_len]
        Quu = Q_t[self.x_len:, self.x_len:]

        Quu_inv = np.linalg.inv(Quu)
        # Quu_inv = np.linalg.inv(Quu + (reg * np.eye(self.u_len)))  # in case singular matrix
        
        qx = q_t[:self.x_len]
        qu = q_t[self.x_len:]

        K_t = -np.matmul(Quu_inv, Qux)
        k_t = -np.matmul(Quu_inv, qu)

        # sanity check
        assert K_t.shape == (self.u_len, self.x_len)
        assert k_t.shape == (self.u_len,)

        Ks[t] = K_t
        ks[t] = k_t

        V_t = Qxx + np.matmul(Qxu, K_t)
        V_t += np.matmul(K_t.transpose(), Qux)
        V_t += np.matmul(np.matmul(K_t.transpose(), Quu), K_t)

        v_t = qx + np.matmul(Qxu, k_t)
        v_t += np.matmul(K_t.transpose(), qu)
        v_t += np.matmul(np.matmul(K_t.transpose(), Quu), k_t)

        # sanity check
        assert V_t.shape == (self.x_len, self.x_len)
        assert v_t.shape == (self.x_len,)

        Vs[t] = V_t
        vs[t] = v_t


    def _ilqg_forward_step(self, transitions_ts, t, Ks, ks):
        # forward pass to build time varying controller
        # namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "t"])
        transitions_t = transitions_ts.get(t, [])
        if len(transitions_t) > 0:
            xs_t = np.vstack([trans.state for trans in transitions_t])
            us_t = np.vstack([trans.action for trans in transitions_t])

            xhat_t = xs_t.mean(axis=0)
            uhat_t = us_t.mean(axis=0)

            assert xhat_t.shape == (self.x_len,)
            assert uhat_t.shape == (self.u_len,)

            ctrl = self._gen_ctrl(xhat_t, uhat_t, Ks[t], ks[t])

        else:
            # default to doing nothing if no transition info is available
            # because we cannot construct any llds or ctrls without that info
            ctrl = self._null_ctrl

        return ctrl

    def _gen_ctrl(self, xhat_t, uhat_t, K_t, k_t):
        # creates a controller, implemented as a closure
        def g(x_t):
            u_t = uhat_t + k_t
            u_t += np.matmul(K_t, (x_t - xhat_t))
            return u_t

        return g

    def _null_ctrl(self, x_t):
        # do nothing
        return np.zeros(self.u_len)
    