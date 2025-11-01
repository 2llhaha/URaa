# pinn_fem_repro_poisson.py
import torch
import yaml
import torch.nn as nn
import os

class PINNFEM(nn.Module):
    def __init__(self, width, g_right, g_left):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 1)
        )
        self.g_right = g_right
        self.g_left = g_left


    def forward(self, x):
    # 确保输入是 (N, 1)
        if x.dim() == 1:
            x = x.unsqueeze(-1)  # (N,) -> (N, 1)
        else :
            x.dim() == 2 and x.shape[1] 
            x = x.view(-1, 1)    # (N, M) -> (N*M, 1)
    
            out = torch.zeros_like(x)
            mask_mid = (x > 0.1) & (x < 0.9)
        
        mask_mid = (x > 0.1) & (x < 0.9)
        mask_right = (x >= 0.9) & (x <= 1.0)
        mask_left = (x >= 0.0) & (x <= 0.1)
    
    # 处理中间区域时，确保 x[mask_mid] 是 (K, 1)
        if mask_mid.any():
            x_mid = x[mask_mid].view(-1, 1)  # 显式重塑
            out[mask_mid] = self.model(x_mid).squeeze()
    
    # 同样处理左右边界
        if mask_right.any():
            x_right = x[mask_right].view(-1, 1)
            out[mask_right] = self.model(x_right) * self.N1(x_right) + self.g_right * self.N2(x_right).squeeze()
         
    
        if mask_left.any():
            x_left = x[mask_left].view(-1, 1)
            out[mask_left] = self.model(x_left) * self.N3(x_left) + self.g_left * self.N4(x_left).squeeze()
    
        return out

    def N1(self, x):
        """"直线，经过点(0.9,1)和(1,0)"""
        return -10 * x + 10
    
    def N2(self, x):
        """直线，经过点(0.9,0)和(1,1)"""
        return 10 * x - 10
    
    def N3(self, x):
        """直线，经过点(0,0)和(0.1,1)"""
        return 10 * x
    
    def N4(self, x):
        """直线，经过点(0,1)和(0.1,0)"""
        return -10 * x + 1

class DataGenerator:
    def __init__(self, config):
        self.config = config
        self.device = config['device']
        
        # 生成并存储各种点集
        self._generate_points()
    
    def _generate_points(self):
        # 内部点 (0.1, 0.9)
        self.interior = torch.linspace(0.1, 0.9, 800).unsqueeze(-1).to(self.device).requires_grad_(True)
        
        # 边界点 (0.0和1.0)
        left_bc = torch.linspace(0.0, 0.1, 100).unsqueeze(-1)
        right_bc = torch.linspace(0.9, 1.0, 100).unsqueeze(-1)
        self.boundary = torch.cat([left_bc, right_bc]).to(self.device)
        
        # 划分训练/验证集
        self._split_datasets()
    
    def _split_datasets(self):
        # 内部点划分
        idx = torch.randperm(len(self.interior))
        split = int(0.8 * len(self.interior))
        self.train_interior = self.interior[idx[:split]]
        self.val_interior = self.interior[idx[split:]]
        
        # 边界点划分
        idx = torch.randperm(len(self.boundary))
        split = int(0.8 * len(self.boundary))
        self.train_boundary = self.boundary[idx[:split]]
        self.val_boundary = self.boundary[idx[split:]]
    
    
    def get_train_points(self):
        return {
        'interior': self.train_interior.reshape(-1, 1),  # 确保 (N, 1)
        'boundary': self.train_boundary.reshape(-1, 1)   # 确保 (M, 1)
        }
    def get_val_points(self):
        return {
            'interior': self.val_interior,
            'boundary': self.val_boundary
        }
    
    
# class Trainer:
#     def __init__(self, model, loss_config, config):
#         self.model = model.to(config['device'])
#         self.loss_config = loss_config
#         self.config = config
#         self.optimizer = torch.optim.Adam(
#             model.parameters(), 
#             lr=config['lr'],
#             weight_decay=1e-4
#         )
#         self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#             self.optimizer, 'min', patience=20, factor=0.5
#         )
#         self.loss_history = []
    
#     def train_step(self, points_dict):
#         self.model.train()
#         self.optimizer.zero_grad()
        
#         # 计算损失
#         total_loss, loss_components = self.loss_config.compute_total_loss(
#             self.model, points_dict
#         )
        
#         # 反向传播
#         total_loss.backward()
#         self.optimizer.step()
        
#         # 记录损失
#         self.loss_history.append({
#             'total': total_loss.item(),
#             **{k: v.item() for k, v in loss_components.items()}
#         })
        
#         return total_loss.item()
    
#     def validate(self, points_dict):
#         self.model.eval()
#         with torch.no_grad():
#             total_loss, loss_components = self.loss_config.compute_total_loss(
#                 self.model, points_dict
#             )
#         return total_loss.item()
    
# class TrainingController:
#     def __init__(self, trainer, data_generator):
#         self.trainer = trainer
#         self.data_handler = data_generator
#         self.best_loss = float('inf')
    
#     def run_training(self, epochs):
#         for epoch in range(1, epochs+1):
#             # 获取训练数据
#             train_points = self.data_handler.get_train_points()
            
#             # 训练步骤
#             train_loss = self.trainer.train_step(train_points)
            
#             # 验证步骤
#             if epoch % 10 == 0:
#                 val_points = self.data_handler.get_val_points()
#                 val_loss = self.trainer.validate(val_points)
#                 self.trainer.scheduler.step(val_loss)
                
#                 # 保存最佳模型
#                 if val_loss < self.best_loss:
#                     self.best_loss = val_loss
#                     torch.save(self.model.state_dict(), 'best_model.pth')
                
#                 # 打印进度
#                 print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")


# #collocation loss function
# class CollocationLoss(nn.Module):
#     def __init__(self, pde, bc_left, bc_right, alpha=1.0):
#         super().__init__()
#         self.pde = pde          # PDE残差函数
#         self.bc_left = bc_left  # 左边界条件函数
#         self.bc_right = bc_right # 右边界条件函数
#         self.alpha = alpha      # 边界损失权重
        
#     def forward(self, model, x_int, x_bc_left, x_bc_right):
#         """
#         x_int: 内部点 (N,1)
#         x_bc_left: 左边界点 (M,1)
#         x_bc_right: 右边界点 (K,1)
#         """
#         # 初始化各损失项
#         losses = {
#             'pde': 0.0,
#             'bc_left': 0.0,
#             'bc_right': 0.0
#         }
        
#         # PDE残差损失（内部点）
#         if x_int is not None:
#             x_int.requires_grad_(True)
#             u_int = model(x_int)
#             pde_res = self.pde(u_int, x_int)
#             losses['pde'] = torch.mean(pde_res**2)
        
#         # 左边界损失
#         if x_bc_left is not None:
#             u_bc_left = model(x_bc_left)
#             bc_left_res = self.bc_left(u_bc_left, x_bc_left)
#             losses['bc_left'] = torch.mean(bc_left_res**2)
        
#         # 右边界损失
#         if x_bc_right is not None:
#             u_bc_right = model(x_bc_right)
#             bc_right_res = self.bc_right(u_bc_right, x_bc_right)
#             losses['bc_right'] = torch.mean(bc_right_res**2)
        
#         # 加权总损失
#         total_loss = losses['pde'] + self.alpha * (losses['bc_left'] + losses['bc_right'])
        
#         return total_loss, losses
   
#     def forward(self, x):
#         x = x.view(-1,1).to(next(self.parameters()).device)
#         out = torch.zeros_like(x)
#         mask_mid = (x > 0.1) & (x < 0.9)
#         mask_right = (x >= 0.9) & (x <= 1.0)
#         mask_left = (x >= 0.0) & (x <= 0.1)
#         if mask_mid.any():
#             out[mask_mid] = self.model(x[mask_mid])
#         if mask_right.any():
#             x_r = x[mask_right]
#             out[mask_right] = self.model(x_r)*self.N1(x_r) + self.g_right*self.N2(x_r)
#         if mask_left.any():
#             x_l = x[mask_left]
#             out[mask_left] = self.model(x_l)*self.N3(x_l) + self.g_left*self.N4(x_l)
#         return out

#     def N1(self,x): return -10*x+10
#     def N2(self,x): return 10*x-10
#     def N3(self,x): return 10*x
#     def N4(self,x): return -10*x+1

# class DataGenerator:
#     def __init__(self, config):
#         self.device = torch.device("cuda" if config.get('device','cpu')=='cuda' and torch.cuda.is_available() else "cpu")
#         self.n_interior = int(config.get('n_interior', 800))
#         self.n_bc_per_side = int(config.get('n_bc_per_side', 100))
#         self._generate_points()

#     def _generate_points(self):
#         self.interior = torch.linspace(0.1, 0.9, self.n_interior, dtype=torch.float32).unsqueeze(-1).to(self.device)
#         left_bc = torch.linspace(0.0, 0.1, self.n_bc_per_side, dtype=torch.float32).unsqueeze(-1)
#         right_bc = torch.linspace(0.9, 1.0, self.n_bc_per_side, dtype=torch.float32).unsqueeze(-1)
#         self.boundary = torch.cat([left_bc, right_bc], dim=0).to(self.device)
#         self._split_datasets()

#     def _split_datasets(self):
#         n = self.interior.size(0)
#         idx = torch.randperm(n, device=self.interior.device)
#         split = int(0.8*n)
#         self.train_interior = self.interior[idx[:split]]
#         self.val_interior = self.interior[idx[split:]]

#         m = self.boundary.size(0)
#         idx_b = torch.randperm(m, device=self.boundary.device)
#         split_b = int(0.8*m)
#         self.train_boundary = self.boundary[idx_b[:split_b]]
#         self.val_boundary = self.boundary[idx_b[split_b:]]

#     def get_train_points(self):
#         return {
#             'interior': self.train_interior.clone().detach(),
#             'boundary': self.train_boundary.clone().detach()
#         }

#     def get_val_points(self):
#         return {
#             'interior': self.val_interior.clone().detach(),
#             'boundary': self.val_boundary.clone().detach()
#         }

class LossConfigurator:
    def __init__(self, pde_fn, bc_fn, device):
        self.pde_fn = pde_fn
        self.bc_fn = bc_fn
        self.device = device

    def compute_total_loss(self, model, points_dict):
        losses = {}
        if 'interior' in points_dict and points_dict['interior'] is not None:
            x_int = points_dict['interior'].to(self.device).requires_grad_(True)
            u_int = model(x_int)
            pde_res = self.pde_fn(model, u_int, x_int)
            losses['pde'] = torch.mean(pde_res**2)
        if 'boundary' in points_dict and points_dict['boundary'] is not None:
            x_bc = points_dict['boundary'].to(self.device)
            u_bc = model(x_bc)
            bc_res = self.bc_fn(model, u_bc, x_bc)
            losses['bc'] = torch.mean(bc_res**2)
        total_loss = sum(losses.values()) if losses else torch.tensor(0.0, device=self.device)
        return total_loss, losses

class Trainer:
    def __init__(self, model, loss_config, config):
        self.model = model.to(loss_config.device)
        self.loss_config = loss_config
        self.config = config
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.get('lr', 1e-3), weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=20, factor=0.5)
        self.loss_history = []

    def train_step(self, points_dict):
        self.model.train()
        self.optimizer.zero_grad()
        total_loss, loss_components = self.loss_config.compute_total_loss(self.model, points_dict)
        total_loss.backward()
        self.optimizer.step()
        record = {'total': total_loss.item()}
        for k,v in loss_components.items():
            record[k] = v.item()
        self.loss_history.append(record)
        return total_loss.item()

    def validate(self, points_dict):
        self.model.eval()
        with torch.no_grad():
            total_loss, _ = self.loss_config.compute_total_loss(self.model, points_dict)
        return total_loss.item()

class TrainingController:
    def __init__(self, trainer, data_generator, save_path="best_model.pth"):
        self.trainer = trainer
        self.data_handler = data_generator
        self.best_loss = float('inf')
        self.save_path = save_path

    def run_training(self, epochs, print_every=10):
        for epoch in range(1, epochs+1):
            train_points = self.data_handler.get_train_points()
            train_loss = self.trainer.train_step(train_points)
            if epoch % print_every == 0 or epoch == 1:
                val_points = self.data_handler.get_val_points()
                val_loss = self.trainer.validate(val_points)
                self.trainer.scheduler.step(val_loss)
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    torch.save(self.trainer.model.state_dict(), self.save_path)
                print(f"Epoch {epoch}: Train Loss={train_loss:.6e}, Val Loss={val_loss:.6e}, Best={self.best_loss:.6e}")

def pde_poisson_fn(model, u, x):
    grad_u = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    grad2_u = torch.autograd.grad(grad_u, x, grad_outputs=torch.ones_like(grad_u), create_graph=True, retain_graph=True)[0]
    f = torch.zeros_like(x)
    return grad2_u - f

def bc_dirichlet_fn(model, u_bc, x_bc):
    x = x_bc.view(-1,1)
    u = u_bc.view(-1,1)
    left_mask = (x <= 0.1).squeeze(-1)
    right_mask = (x >= 0.9).squeeze(-1)
    res = torch.zeros_like(u)
    if left_mask.any():
        g_left = model.g_left if hasattr(model, 'g_left') else torch.tensor(0.0, device=u.device)
        res[left_mask] = u[left_mask] - g_left
    if right_mask.any():
        g_right = model.g_right if hasattr(model, 'g_right') else torch.tensor(0.0, device=u.device)
        res[right_mask] = u[right_mask] - g_right
    return res

if __name__=="__main__":
    defaults = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'epochs': 500,
        'lr': 1e-3,
        'width': 64,
        'g_left': 0.0,
        'g_right': 0.0
    }
    cfg_path = "C:/Users/26303/Desktop/0809_hybridV1/config.yaml"
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r', encoding='utf-8') as f:
            file_cfg = yaml.safe_load(f) or {}
    else:
        file_cfg = {}
    cfg = {**defaults, **file_cfg}
    device = 'cuda' if cfg.get('device')=='cuda' and torch.cuda.is_available() else 'cpu'
    cfg['device'] = device

    data_gen = DataGenerator(cfg)
    model = PINNFEM(int(cfg['width']), cfg.get('g_right', 0.0), cfg.get('g_left', 0.0)).to(device)

    train_pts = data_gen.get_train_points()
    print("train_interior shape:", train_pts['interior'].shape)
    print("train_boundary shape:", train_pts['boundary'].shape)

    loss_conf = LossConfigurator(pde_poisson_fn, bc_dirichlet_fn, device=device)
    trainer = Trainer(model, loss_conf, cfg)
    controller = TrainingController(trainer, data_gen, save_path="best_model.pth")
    controller.run_training(int(cfg.get('epochs', 500)), print_every=10)
