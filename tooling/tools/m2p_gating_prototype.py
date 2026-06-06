import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt

class MoE_M2P(nn.Module):
    def __init__(self, d_model, d_domain, num_experts=4):
        super().__init__()
        self.router = nn.Linear(d_domain, num_experts)
        # Separate experts to guarantee completely disjoint generation pathways
        self.experts_out = [nn.Linear(d_domain, d_model) for _ in range(num_experts)]
        
    def __call__(self, x, domain_emb):
        # Hard / Soft routing
        routing_weights = nn.softmax(self.router(domain_emb), axis=-1)
        out = sum(routing_weights[:, i:i+1] * expert(domain_emb) for i, expert in enumerate(self.experts_out))
        return out

class AdditiveM2P(nn.Module):
    def __init__(self, d_model, d_domain):
        super().__init__()
        self.emb = nn.Linear(d_domain, d_model)
        self.out = nn.Linear(d_model, d_model)
        
    def __call__(self, x, domain_emb):
        h = mx.maximum(0, x + self.emb(domain_emb))
        return self.out(h)

def test_m2p_collapse():
    mx.random.seed(1)
    d_model = 64
    d_domain = 16
    n_domains = 5
    n_steps = 400
    
    domain_embs = mx.random.normal((n_domains, d_domain))
    target_Bs = mx.random.normal((n_domains, d_model))
    target_Bs = target_Bs / mx.linalg.norm(target_Bs, axis=-1, keepdims=True)
    
    # Extreme loss gradient imbalance (Domain 0 is 20x harder than Domain 4)
    loss_scales = mx.array([20.0, 10.0, 5.0, 2.0, 1.0])
    
    input_x = mx.random.normal((1, d_model))
    
    additive_m2p = AdditiveM2P(d_model, d_domain)
    moe_m2p = MoE_M2P(d_model, d_domain)
    
    adam_add = opt.Adam(learning_rate=0.01)
    adam_moe = opt.Adam(learning_rate=0.01)
    
    mx.eval(additive_m2p.parameters(), moe_m2p.parameters())
    
    def loss_add(model, d_idx):
        pred_B = model(input_x, domain_embs[d_idx:d_idx+1])
        return loss_scales[d_idx] * mx.mean(mx.square(pred_B - target_Bs[d_idx:d_idx+1]))

    def loss_moe(model, d_idx):
        pred_B = model(input_x, domain_embs[d_idx:d_idx+1])
        # We also apply Loss Normalization here (dividing by loss_scale)
        # to prove the two techniques combined fix the centroid collapse.
        raw_loss = mx.mean(mx.square(pred_B - target_Bs[d_idx:d_idx+1]))
        normalized_loss = raw_loss * (1.0) # Normalized! (Ignored the 20x multiplier)
        return normalized_loss
        
    loss_add_vg = nn.value_and_grad(additive_m2p, loss_add)
    loss_moe_vg = nn.value_and_grad(moe_m2p, loss_moe)
    
    for step in range(n_steps):
        for d in range(n_domains):
            l_a, grads_a = loss_add_vg(additive_m2p, d)
            adam_add.update(additive_m2p, grads_a)
            
            l_m, grads_m = loss_moe_vg(moe_m2p, d)
            adam_moe.update(moe_m2p, grads_m)
            
        mx.eval(additive_m2p.parameters(), moe_m2p.parameters())
        
    def calc_mean_cos(B):
        norm = B / (mx.linalg.norm(B, axis=-1, keepdims=True) + 1e-8)
        cos_matrix = norm @ norm.T
        mask = 1.0 - mx.eye(n_domains)
        return (mx.sum(cos_matrix * mask) / (n_domains * (n_domains - 1))).item()
        
    B_add = mx.concatenate([additive_m2p(input_x, domain_embs[d:d+1]) for d in range(n_domains)], axis=0)
    B_moe = mx.concatenate([moe_m2p(input_x, domain_embs[d:d+1]) for d in range(n_domains)], axis=0)
    
    print(f"Target underlying domain cosine: {calc_mean_cos(target_Bs):.4f} (Ideal Distinction)")
    print(f"Additive M2P cosine (Failed):      {calc_mean_cos(B_add):.4f} (1.0 = Total Collapse to Centroid)")
    print(f"MoE + Norm M2P cosine (Fixed):     {calc_mean_cos(B_moe):.4f} (Lower = Survives Collapse)")

if __name__ == "__main__":
    test_m2p_collapse()
