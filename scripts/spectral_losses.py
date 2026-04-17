import torch
import torch.nn as nn
import torch.nn.functional as F


def first_derivative(x: torch.Tensor) -> torch.Tensor:
    return x[..., 1:] - x[..., :-1]


def second_derivative(x: torch.Tensor) -> torch.Tensor:
    d1 = first_derivative(x)
    return d1[..., 1:] - d1[..., :-1]


def total_variation(x: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(first_derivative(x)))


def _masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum().clamp_min(1.0)
    return torch.sum(torch.abs(pred - target) * mask) / denom


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum().clamp_min(1.0)
    return torch.sum(((pred - target) ** 2) * mask) / denom


def _weighted_l1(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    denom = weights.sum().clamp_min(1.0)
    return torch.sum(torch.abs(pred - target) * weights) / denom


def _weighted_mse(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    denom = weights.sum().clamp_min(1.0)
    return torch.sum(((pred - target) ** 2) * weights) / denom


def _smoothed_signal(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size <= 1:
        return x
    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2
    x_pad = F.pad(x, (pad, pad), mode="reflect")
    return F.avg_pool1d(x_pad, kernel_size=kernel_size, stride=1)


class PeakAlignmentLoss(nn.Module):
    """Penalize prediction errors around derivative zero-crossings."""

    def __init__(self, window: int = 5, weight: float = 2.0):
        super().__init__()
        self.window = max(1, int(window))
        self.weight = float(weight)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        supervision_mask: torch.Tensor | None = None,
        region_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        d1_target = first_derivative(target)
        sign_change = (d1_target[..., :-1] * d1_target[..., 1:]) < 0
        mask = sign_change.float()
        if self.window > 1:
            pad = self.window // 2
            mask = F.max_pool1d(
                F.pad(mask, (pad, pad), mode="replicate"),
                kernel_size=self.window,
                stride=1,
            )
        mask = F.pad(mask, (1, 1), mode="replicate")
        weights = 1.0 + self.weight * mask
        if region_mask is not None:
            weights = weights * region_mask
        if supervision_mask is not None:
            weights = weights * supervision_mask
        return _weighted_l1(pred, target, weights)


class CompositeSpectralLoss(nn.Module):
    def __init__(
        self,
        w_mse: float = 0.50,
        w_l1: float = 1.0,
        w_d1: float = 0.15,
        w_d2: float = 0.10,
        w_tv: float = 0.0025,
        w_fft: float = 0.05,
        w_amp: float = 0.10,
        w_baseline: float = 0.05,
        w_peak_profile: float = 0.18,
        w_peak_center: float = 0.30,
        w_edge_l1: float = 0.18,
        w_edge_d1: float = 0.16,
        edge_points: int = 96,
        peak_window_radius: int = 8,
        peak_quantile: float = 0.86,
        peak_softmax_temp: float = 12.0,
        mask_leading_points: int = 0,
        w_smooth_consistency: float = 0.0,
        smooth_kernel_size: int = 9,
        w_fingerprint_l1: float = 0.0,
        w_fingerprint_mse: float = 0.0,
        w_fingerprint_d1: float = 0.0,
        fingerprint_start: int = 104,
        fingerprint_end: int = 726,
        w_curvature_mse: float = 0.0,
        curvature_scale: float = 12.0,
        w_valley_under: float = 0.0,
        valley_quantile: float = 0.20,
        w_valley_center: float = 0.0,
        w_peak_align: float = 0.0,
        peak_align_window: int = 5,
        peak_align_weight: float = 2.0,
        peak_point_weight: float = 1.25,
        slope_point_weight: float = 0.75,
        fingerprint_point_weight: float = 0.0,
        peak_focus_start: int = -1,
        peak_focus_end: int = -1,
        w_peak_dominance: float = 0.0,
        peak_dominance_margin_scale: float = 0.25,
    ):
        super().__init__()
        self.w_mse = w_mse
        self.w_l1 = w_l1
        self.w_d1 = w_d1
        self.w_d2 = w_d2
        self.w_tv = w_tv
        self.w_fft = w_fft
        self.w_amp = w_amp
        self.w_baseline = w_baseline
        self.w_peak_profile = w_peak_profile
        self.w_peak_center = w_peak_center
        self.w_edge_l1 = w_edge_l1
        self.w_edge_d1 = w_edge_d1
        self.edge_points = edge_points
        self.peak_window_radius = peak_window_radius
        self.peak_quantile = peak_quantile
        self.peak_softmax_temp = peak_softmax_temp
        self.mask_leading_points = mask_leading_points
        self.w_smooth_consistency = w_smooth_consistency
        self.smooth_kernel_size = smooth_kernel_size
        self.w_fingerprint_l1 = w_fingerprint_l1
        self.w_fingerprint_mse = w_fingerprint_mse
        self.w_fingerprint_d1 = w_fingerprint_d1
        self.fingerprint_start = fingerprint_start
        self.fingerprint_end = fingerprint_end
        self.w_curvature_mse = w_curvature_mse
        self.curvature_scale = curvature_scale
        self.w_valley_under = w_valley_under
        self.valley_quantile = valley_quantile
        self.w_valley_center = w_valley_center
        self.w_peak_align = w_peak_align
        self.low_boundary_points = 64
        self.boundary_weight = 2.5
        self.peak_weight = peak_point_weight
        self.slope_weight = slope_point_weight
        self.fingerprint_point_weight = fingerprint_point_weight
        self.peak_focus_start = peak_focus_start
        self.peak_focus_end = peak_focus_end
        self.w_peak_dominance = w_peak_dominance
        self.peak_dominance_margin_scale = peak_dominance_margin_scale
        self.peak_alignment = PeakAlignmentLoss(window=peak_align_window, weight=peak_align_weight)

    def _supervision_mask(self, target: torch.Tensor) -> torch.Tensor:
        mask = torch.ones_like(target)
        if self.mask_leading_points > 0:
            mask[..., : min(self.mask_leading_points, target.size(-1))] = 0.0
        return mask

    def _point_weights(self, target: torch.Tensor) -> torch.Tensor:
        weights = torch.ones_like(target)

        boundary_points = min(self.low_boundary_points, target.size(-1))
        weights[..., :boundary_points] += self.boundary_weight

        target_abs = torch.abs(target)
        peak_threshold = torch.quantile(target_abs.view(target.size(0), -1), 0.80, dim=1)
        peak_mask = (target_abs >= peak_threshold.view(-1, 1, 1)).float()
        weights = weights + self.peak_weight * peak_mask

        slopes = torch.abs(first_derivative(target))
        slope_threshold = torch.quantile(slopes.view(slopes.size(0), -1), 0.80, dim=1)
        slope_mask = (slopes >= slope_threshold.view(-1, 1, 1)).float()
        slope_mask = F.pad(slope_mask, (1, 0), mode="replicate")
        weights = weights + self.slope_weight * slope_mask

        fp_start = max(0, min(self.fingerprint_start, target.size(-1) - 1))
        fp_end = max(fp_start + 1, min(self.fingerprint_end, target.size(-1)))
        if self.fingerprint_point_weight > 0.0 and fp_end > fp_start:
            weights[..., fp_start:fp_end] += self.fingerprint_point_weight
        return weights

    def _focus_mask(self, target: torch.Tensor) -> torch.Tensor:
        if self.peak_focus_start < 0 or self.peak_focus_end <= self.peak_focus_start:
            return torch.ones_like(target)
        start = max(0, min(self.peak_focus_start, target.size(-1) - 1))
        end = max(start + 1, min(self.peak_focus_end, target.size(-1)))
        mask = torch.zeros_like(target)
        mask[..., start:end] = 1.0
        return mask

    def _curvature_weighted_mse(self, pred: torch.Tensor, target: torch.Tensor, region_mask: torch.Tensor) -> torch.Tensor:
        d2_target = torch.abs(second_derivative(target))
        d2_target = F.pad(d2_target, (1, 1), mode="replicate")
        masked_d2 = d2_target * region_mask
        flat = masked_d2.view(masked_d2.size(0), -1)
        max_vals = flat.max(dim=1).values.view(-1, 1, 1).clamp_min(1e-6)
        norm_curvature = masked_d2 / max_vals
        weights = (1.0 + self.curvature_scale * norm_curvature) * region_mask
        return _weighted_mse(pred, target, weights)

    def _peak_windows_and_center_loss(self, pred: torch.Tensor, target: torch.Tensor, focus_mask: torch.Tensor):
        radius = self.peak_window_radius
        kernel = radius * 2 + 1
        target_abs = torch.abs(target) * focus_mask
        pooled = F.max_pool1d(target_abs, kernel_size=kernel, stride=1, padding=radius)
        threshold = torch.quantile(target_abs.view(target.size(0), -1), self.peak_quantile, dim=1).view(-1, 1, 1)
        peak_centers = ((target_abs >= pooled - 1e-7) & (target_abs >= threshold)).float()
        peak_windows = F.max_pool1d(peak_centers, kernel_size=kernel, stride=1, padding=radius) * focus_mask

        center_total = pred.new_tensor(0.0)
        center_count = 0
        dominance_total = pred.new_tensor(0.0)
        dominance_count = 0
        offset_grid = torch.arange(-radius, radius + 1, device=pred.device, dtype=pred.dtype)
        window_len = offset_grid.numel()
        target_abs_flat = target_abs[:, 0]
        pred_abs_flat = torch.abs(pred)[:, 0]

        for batch_idx in range(pred.size(0)):
            center_indices = torch.nonzero(peak_centers[batch_idx, 0] > 0.5, as_tuple=False).flatten().tolist()
            if not center_indices:
                continue

            pruned_centers = []
            last_kept = -10 * kernel
            for center_idx in center_indices:
                if center_idx - last_kept > radius:
                    pruned_centers.append(center_idx)
                    last_kept = center_idx

            for center_idx in pruned_centers:
                left = max(0, center_idx - radius)
                right = min(target.size(-1), center_idx + radius + 1)
                pad_left = max(0, radius - center_idx)
                pad_right = max(0, center_idx + radius + 1 - target.size(-1))

                pred_window = F.pad(pred_abs_flat[batch_idx, left:right], (pad_left, pad_right))
                target_window = F.pad(target_abs_flat[batch_idx, left:right], (pad_left, pad_right))
                if pred_window.numel() != window_len or target_window.numel() != window_len:
                    continue

                pred_prob = torch.softmax(pred_window * self.peak_softmax_temp, dim=-1)
                target_prob = torch.softmax(target_window * self.peak_softmax_temp, dim=-1)
                pred_center = torch.sum(pred_prob * offset_grid)
                target_center = torch.sum(target_prob * offset_grid)
                center_total = center_total + torch.abs(pred_center - target_center) / max(radius, 1)
                center_count += 1

                center_slot = radius
                pred_center_val = pred_window[center_slot]
                target_center_val = target_window[center_slot]
                target_margin = (target_center_val - target_window).clamp_min(0.0)
                if self.peak_dominance_margin_scale > 0.0:
                    target_margin = target_margin * self.peak_dominance_margin_scale
                valid_neighbors = torch.ones_like(target_margin, dtype=torch.bool)
                valid_neighbors[center_slot] = False
                valid_neighbors &= target_margin > 0
                if valid_neighbors.any():
                    dominance_penalty = F.relu(pred_window[valid_neighbors] - pred_center_val + target_margin[valid_neighbors])
                    dominance_total = dominance_total + dominance_penalty.mean()
                    dominance_count += 1

        if center_count == 0:
            center_loss = pred.new_tensor(0.0)
        else:
            center_loss = center_total / center_count

        if dominance_count == 0:
            dominance_loss = pred.new_tensor(0.0)
        else:
            dominance_loss = dominance_total / dominance_count

        return peak_windows, center_loss, dominance_loss

    def _valley_windows_and_center_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        focus_mask: torch.Tensor,
        quantile: float,
    ):
        radius = self.peak_window_radius
        kernel = radius * 2 + 1
        target_signal = F.relu(-target) * focus_mask
        pooled = F.max_pool1d(target_signal, kernel_size=kernel, stride=1, padding=radius)
        threshold = torch.quantile(target_signal.view(target.size(0), -1), quantile, dim=1).view(-1, 1, 1)
        valley_centers = ((target_signal >= pooled - 1e-7) & (target_signal >= threshold)).float()
        valley_windows = F.max_pool1d(valley_centers, kernel_size=kernel, stride=1, padding=radius) * focus_mask

        center_total = pred.new_tensor(0.0)
        center_count = 0
        offset_grid = torch.arange(-radius, radius + 1, device=pred.device, dtype=pred.dtype)
        window_len = offset_grid.numel()
        target_flat = target_signal[:, 0]
        pred_flat = F.relu(-pred)[:, 0]

        for batch_idx in range(pred.size(0)):
            center_indices = torch.nonzero(valley_centers[batch_idx, 0] > 0.5, as_tuple=False).flatten().tolist()
            if not center_indices:
                continue

            pruned_centers = []
            last_kept = -10 * kernel
            for center_idx in center_indices:
                if center_idx - last_kept > radius:
                    pruned_centers.append(center_idx)
                    last_kept = center_idx

            for center_idx in pruned_centers:
                left = max(0, center_idx - radius)
                right = min(target.size(-1), center_idx + radius + 1)
                pad_left = max(0, radius - center_idx)
                pad_right = max(0, center_idx + radius + 1 - target.size(-1))

                pred_window = F.pad(pred_flat[batch_idx, left:right], (pad_left, pad_right))
                target_window = F.pad(target_flat[batch_idx, left:right], (pad_left, pad_right))
                if pred_window.numel() != window_len or target_window.numel() != window_len:
                    continue

                pred_prob = torch.softmax(pred_window * self.peak_softmax_temp, dim=-1)
                target_prob = torch.softmax(target_window * self.peak_softmax_temp, dim=-1)
                pred_center = torch.sum(pred_prob * offset_grid)
                target_center = torch.sum(target_prob * offset_grid)
                center_total = center_total + torch.abs(pred_center - target_center) / max(radius, 1)
                center_count += 1

        if center_count == 0:
            center_loss = pred.new_tensor(0.0)
        else:
            center_loss = center_total / center_count

        return valley_windows, center_loss

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        supervision_mask = self._supervision_mask(target)
        point_weights = self._point_weights(target) * supervision_mask
        d1_mask = supervision_mask[..., 1:] * supervision_mask[..., :-1]
        d2_mask = supervision_mask[..., 2:] * supervision_mask[..., 1:-1] * supervision_mask[..., :-2]
        d1_weights = point_weights[..., 1:] * d1_mask
        d2_weights = point_weights[..., 2:] * d2_mask

        mse = torch.sum(((pred - target) ** 2) * supervision_mask) / supervision_mask.sum().clamp_min(1.0)
        l1 = _weighted_l1(pred, target, point_weights)
        d1 = _weighted_mse(first_derivative(pred), first_derivative(target), d1_weights)
        d2 = _weighted_mse(second_derivative(pred), second_derivative(target), d2_weights)

        pred_fft = torch.fft.rfft(pred.float(), dim=-1)
        target_fft = torch.fft.rfft(target.float(), dim=-1)
        fft = F.l1_loss(torch.log1p(torch.abs(pred_fft)), torch.log1p(torch.abs(target_fft)))

        target_abs = torch.abs(target)
        peak_threshold = torch.quantile(target_abs.view(target.size(0), -1), 0.80, dim=1)
        peak_mask = (target_abs >= peak_threshold.view(-1, 1, 1)).float() * supervision_mask
        amplitude = _masked_l1(pred, target, peak_mask)
        focus_mask = self._focus_mask(target) * supervision_mask
        peak_windows, peak_center, peak_dominance = self._peak_windows_and_center_loss(pred, target, focus_mask)
        peak_profile = _masked_l1(pred, target, peak_windows * supervision_mask)

        baseline_threshold = torch.quantile(target_abs.view(target.size(0), -1), 0.35, dim=1)
        baseline_mask = (target_abs <= baseline_threshold.view(-1, 1, 1)).float() * supervision_mask
        baseline = _masked_l1(pred, target, baseline_mask)
        smooth_consistency = _masked_mse(pred, _smoothed_signal(pred, self.smooth_kernel_size), baseline_mask)

        edge_points = min(self.edge_points, target.size(-1))
        if edge_points > 1:
            edge_mask = supervision_mask[..., :edge_points]
            edge_pred = pred[..., :edge_points]
            edge_target = target[..., :edge_points]
            edge_l1 = _weighted_l1(edge_pred, edge_target, edge_mask)
            edge_d1_mask = edge_mask[..., 1:] * edge_mask[..., :-1]
            edge_d1 = _weighted_l1(first_derivative(edge_pred), first_derivative(edge_target), edge_d1_mask)
        else:
            edge_l1 = pred.new_tensor(0.0)
            edge_d1 = pred.new_tensor(0.0)

        fp_start = max(0, min(self.fingerprint_start, target.size(-1) - 1))
        fp_end = max(fp_start + 1, min(self.fingerprint_end, target.size(-1)))
        fp_mask = supervision_mask[..., fp_start:fp_end]
        fp_pred = pred[..., fp_start:fp_end]
        fp_target = target[..., fp_start:fp_end]
        fingerprint_l1 = _weighted_l1(fp_pred, fp_target, fp_mask)
        fingerprint_mse = _masked_mse(fp_pred, fp_target, fp_mask)
        curvature_mse = self._curvature_weighted_mse(pred, target, focus_mask)
        fp_d1_mask = fp_mask[..., 1:] * fp_mask[..., :-1]
        fingerprint_d1 = _weighted_l1(first_derivative(fp_pred), first_derivative(fp_target), fp_d1_mask)
        valley_threshold = torch.quantile(fp_target.view(fp_target.size(0), -1), self.valley_quantile, dim=1).view(-1, 1, 1)
        valley_mask = (fp_target <= valley_threshold).float() * fp_mask
        valley_under = torch.sum(F.relu(fp_pred - fp_target) * valley_mask) / valley_mask.sum().clamp_min(1.0)
        valley_windows, valley_center = self._valley_windows_and_center_loss(pred, target, focus_mask, self.valley_quantile)
        peak_align = self.peak_alignment(pred, target, supervision_mask, focus_mask)

        tv = total_variation(pred)

        return (
            self.w_mse * mse
            + self.w_l1 * l1
            + self.w_d1 * d1
            + self.w_d2 * d2
            + self.w_tv * tv
            + self.w_fft * fft
            + self.w_amp * amplitude
            + self.w_baseline * baseline
            + self.w_smooth_consistency * smooth_consistency
            + self.w_peak_profile * peak_profile
            + self.w_peak_center * peak_center
            + self.w_peak_dominance * peak_dominance
            + self.w_edge_l1 * edge_l1
            + self.w_edge_d1 * edge_d1
            + self.w_fingerprint_l1 * fingerprint_l1
            + self.w_fingerprint_mse * fingerprint_mse
            + self.w_curvature_mse * curvature_mse
            + self.w_fingerprint_d1 * fingerprint_d1
            + self.w_valley_under * valley_under
            + self.w_valley_center * valley_center
            + self.w_peak_align * peak_align
        )
