import numpy as np
import torch


class SafetyMetrics:

    
    def __init__(self, ttc_crit=1.5, max_decel=3.3, 
                 w_ttc=1.0, w_sth=0.5, w_msfd=5.0,
                 lambda_ttc=1.0, gamma_sth=1.0,
                 epsilon=0.001):

        self.ttc_crit = ttc_crit
        self.max_decel = max_decel
        self.w_ttc = w_ttc
        self.w_sth = w_sth
        self.w_msfd = w_msfd
        self.lambda_ttc = lambda_ttc
        self.gamma_sth = gamma_sth
        self.epsilon = epsilon
    
    def compute_ttc(self, distance, ego_vel, lead_vel):

        rel_vel = max(ego_vel - lead_vel, 0)

        ttc = distance / (rel_vel + self.epsilon)
        
        return ttc
    
    def compute_thw(self, distance, ego_vel):

        thw = distance / (ego_vel + self.epsilon)
        
        return thw
    
    def compute_sth(self, distance, ego_vel, lead_vel):

        thw = self.compute_thw(distance, ego_vel)
        

        denominator = 1 - (ego_vel - lead_vel) / (3.3 * self.max_decel)

        if denominator <= 0:
            return 10.0  
        
        sth = thw / denominator
        
        return sth
    
    def compute_msfd(self, ego_vel):

        msfd = (ego_vel ** 2) / (2 * self.max_decel)
        
        return msfd
    
    def compute_reward(self, distance, ego_vel, lead_vel):

        ttc = self.compute_ttc(distance, ego_vel, lead_vel)
        thw = self.compute_thw(distance, ego_vel)
        sth = self.compute_sth(distance, ego_vel, lead_vel)
        msfd = self.compute_msfd(ego_vel)

        ttc_reward = self.w_ttc * np.exp(-self.lambda_ttc * (ttc - self.ttc_crit) ** 2)
        

        sth_reward = self.w_sth * np.exp(-self.gamma_sth * (thw - sth) ** 2)
        
        # MSFD惩罚: 当距离小于MSFD时给予惩罚
        msfd_penalty = self.w_msfd * float(distance < msfd)

        total_reward = ttc_reward + sth_reward - msfd_penalty
        
        return total_reward, {
            'ttc': ttc,
            'thw': thw,
            'sth': sth,
            'msfd': msfd,
            'ttc_reward': ttc_reward,
            'sth_reward': sth_reward,
            'msfd_penalty': msfd_penalty
        }