    def render_with_refraction(self, rays_o, rays_d, near, far, perturb_overwrite=-1, background_rgb=None, cos_anneal_ratio=0.0):
        """
        ReNeuS rendering with refraction-aware ray tracing.
        Simplified version: single refraction entry into container, then standard NeuS rendering.
        
        Args:
            rays_o: [batch_size, 3] camera ray origins
            rays_d: [batch_size, 3] camera ray directions
            near, far: near and far bounds
            background_rgb: background color
            cos_anneal_ratio: cosine annealing ratio for NeuS
        
        Returns:
            Dictionary with rendering results (same format as original render)
        """
        batch_size = rays_o.shape[0]
        device = rays_o.device
        
        # Step 1: Compute intersection with container surface
        hit_mask, hit_points, hit_normals, hit_distances = ray_mesh_intersection(
            rays_o, rays_d, self.ray_tracer
        )
        
        # Initialize output
        color_fine = torch.zeros(batch_size, 3, device=device)
        s_val = torch.zeros(batch_size, 1, device=device)
        weight_sum = torch.zeros(batch_size, 1, device=device)
        gradients_list = []
        gradient_error_sum = 0.0
        
        # Step 2: Handle rays that miss the container (render as background)
        if background_rgb is not None:
            color_fine[~hit_mask] = background_rgb
        
        if not hit_mask.any():
            # All rays missed, return background
            return {
                'color_fine': color_fine,
                's_val': s_val,
                'weight_sum': weight_sum,
                'gradient_error': torch.tensor(0.0, device=device),
                'weights': torch.zeros(batch_size, self.n_samples, device=device),
                'gradients': torch.zeros(batch_size, self.n_samples, 3, device=device)
            }
        
        # Step 3: For rays hitting the container, compute refraction
        hit_indices = torch.where(hit_mask)[0]
        rays_o_hit = rays_o[hit_indices]
        rays_d_hit = rays_d[hit_indices]
        hit_pts = hit_points[hit_indices]
        hit_norms = hit_normals[hit_indices]
        
        # Compute refracted ray direction (air -> container)
        rays_d_refract, valid_refract = compute_refraction(
            rays_d_hit, hit_norms, ior_in=1.0, ior_out=self.ior
        )
        
        # Small offset to avoid self-intersection
        eps = 1e-4
        rays_o_refract = hit_pts + eps * rays_d_refract
        
        # Step 4: Render along refracted rays using original NeuS logic
        # We need to compute new near/far bounds for the refracted rays
        # Simple approximation: use distance-based bounds
        near_refract = torch.zeros(hit_indices.shape[0], device=device) + 0.0
        far_refract = torch.zeros(hit_indices.shape[0], device=device) + 2.0
        
        # For simplicity, call render_core directly for refracted rays
        # This is a simplified implementation - full version would handle multiple bounces
        
        batch_size_hit = hit_indices.shape[0]
        sample_dist = 2.0 / self.n_samples
        z_vals = torch.linspace(0.0, 1.0, self.n_samples, device=device)
        z_vals = near_refract[:, None] + (far_refract[:, None] - near_refract[:, None]) * z_vals[None, :]
        
        # Perturb
        perturb = self.perturb if perturb_overwrite < 0 else perturb_overwrite
        if perturb > 0:
            t_rand = (torch.rand([batch_size_hit, 1], device=device) - 0.5)
            z_vals = z_vals + t_rand * 2.0 / self.n_samples
        
        # Up-sample (same as original NeuS)
        if self.n_importance > 0:
            with torch.no_grad():
                pts = rays_o_refract[:, None, :] + rays_d_refract[:, None, :] * z_vals[..., :, None]
                sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size_hit, self.n_samples)
                
                for i in range(self.up_sample_steps):
                    new_z_vals = self.up_sample(
                        rays_o_refract, rays_d_refract, z_vals, sdf,
                        self.n_importance // self.up_sample_steps, 64 * 2**i
                    )
                    z_vals, sdf = self.cat_z_vals(
                        rays_o_refract, rays_d_refract, z_vals, new_z_vals, sdf,
                        last=(i + 1 == self.up_sample_steps)
                    )
        
        # Render core
        ret_fine = self.render_core(
            rays_o_refract, rays_d_refract, z_vals, sample_dist,
            self.sdf_network, self.deviation_network, self.color_network,
            background_rgb=background_rgb,
            cos_anneal_ratio=cos_anneal_ratio
        )
        
        # Fill in results for hit rays
        color_fine[hit_indices] = ret_fine['color']
        weight_sum[hit_indices] = ret_fine['weights'].sum(dim=-1, keepdim=True)
        
        n_samples_final = z_vals.shape[1]
        s_val_mean = ret_fine['s_val'].reshape(batch_size_hit, n_samples_final).mean(dim=-1, keepdim=True)
        s_val[hit_indices] = s_val_mean
        
        # Collect gradients
        gradients_all = torch.zeros(batch_size, n_samples_final, 3, device=device)
        gradients_all[hit_indices] = ret_fine['gradients']
        
        return {
            'color_fine': color_fine,
            's_val': s_val,
            'weight_sum': weight_sum,
            'weight_max': weight_sum,  # Simplified
            'gradient_error': ret_fine['gradient_error'],
            'gradients': gradients_all,
            'weights': torch.zeros(batch_size, n_samples_final, device=device).scatter_(
                0, hit_indices[:, None].expand(-1, n_samples_final),
                ret_fine['weights']
            ) if hit_mask.any() else torch.zeros(batch_size, self.n_samples, device=device),
        }

