import torch
import torch.nn as nn
import torch.nn.functional as F


class Wrapper(nn.Module):
    def __init__(self, student, teacher, is_clip=False):
        super(Wrapper, self).__init__()
        if not is_clip:
            self.t_dims = teacher.fc.in_features
            self.s_dims = student.fc.in_features
            teacher.fc = nn.Identity()
            student.fc = nn.Identity()
        else:
            self.t_dims = 1024
            self.s_dims = 1024

        #self.ln_post = nn.LayerNorm(width)
        self.backbone = student
        print(self.t_dims, self.s_dims, 'teacher-student hidden dims. and use_proj')
        if self.t_dims != self.s_dims:
            self.proj_head = nn.Sequential(
                nn.Linear(self.s_dims, 2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, self.t_dims)
            )
        else:
            print('SKIPPING PROJECTION HEAD')
            self.proj_head = None

    def forward(self, x):
        last_feat = self.backbone(x)#.view(-1, self.s_dims)  # Identity allows to get last features
        if self.proj_head is not None:
            last_feat = self.proj_head(last_feat)
        return last_feat
