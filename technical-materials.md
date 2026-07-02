---
layout: default
title: "Technical Materials"
description: "Supporting code examples, diagrams, and implementation details"
---

# Technical Materials: Infrastructure → ML Systems
## Supporting Code & Diagrams for the Three-Page Journey

This collection provides detailed implementation examples that support the technical writeup's demonstration of how infrastructure engineering expertise naturally extends to production ML systems. Each section corresponds to one of the three pages in the main writeup, showing the progression from foundational infrastructure through advanced ML engineering to real-world production operations.

### Materials Organization
- **Part 1: Infrastructure Foundations** - Supporting [Page 1: Container Orchestration](README#page-1-container-orchestration--service-mesh-architecture)
- **Part 2: ML Pipeline Implementation** - Supporting [Page 2: ML Engineering](README#page-2-ml-pipeline-engineering-deep-dive)  
- **Part 3: Production Operations** - Supporting [Page 3: Operations & Web Platform](README#page-3-production-operations--web-platform)

---

## Part 1: Infrastructure Foundations
*Supporting Page 1: Container Orchestration & Service Mesh Architecture*

These examples demonstrate the infrastructure engineering patterns that provide the reliability foundation for production ML workloads. The cost optimization strategies and container orchestration patterns shown here are the same distributed systems principles that enable scalable AI development.

## Cost Optimization Strategy Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    MULTI-TIER COST STRATEGY                     │
└─────────────────────────────────────────────────────────────────┘

  High Performance ────────────────────────► High Cost
       │                                        │
       ▼                                        ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ G6.xlarge   │────▶│ G5.xlarge   │────▶│G4dn.xlarge  │
│ $1.61/hr    │     │ ~$0.48/hr   │     │ ~$0.39/hr   │
│ On-Demand   │     │ Spot (70%↓) │     │ Spot (76%↓) │
│ Primary     │     │ Fallback #1 │     │ Fallback #2 │
└─────────────┘     └─────────────┘     └─────────────┘
       │                   │                   │
       ▼                   ▼                   ▼
   Reliability         Performance         Availability
   Guarantee           Optimization        Diversification

┌─────────────────────────────────────────────────────────────────┐
│                     LIGHTWEIGHT SERVICES                        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Discord Bot │     │   Mecene    │     │  Web App    │
│ 159MB/0.17  │     │ 159MB/0.17  │     │ 1GB/1vCPU   │
│    vCPU     │     │    vCPU     │     │             │
│~$0.003/hr   │     │~$0.003/hr   │     │~$0.02/hr    │
└─────────────┘     └─────────────┘     └─────────────┘

Total Monthly Cost Optimization: ~75% savings vs. all on-demand
```

---

## Part 2: ML Pipeline Implementation
*Supporting Page 2: ML Pipeline Engineering Deep Dive*

The following code examples showcase advanced ML engineering techniques that go beyond standard framework usage. These implementations demonstrate how infrastructure engineering thinking—modular design, optimization, observability—applies directly to building sophisticated AI systems.

### Two-Stage Generation Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 1: TEXT-ONLY BASE                      │
└─────────────────────────────────────────────────────────────────┘

Input: "professional portrait, corporate confidence"
         │
         ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Text Encoding  │───▶│ SDXL Diffusion  │───▶│  Base Image     │
│ CLIP Embeddings│    │Custom Scheduler │    │ 512x512 RGB     │
│[2, 77, 2048]   │    │20 steps, CFG=8  │    │ Generic Face    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                    start_merge_step = 1000 (disabled)
                    Face conditioning: NEVER APPLIED

┌─────────────────────────────────────────────────────────────────┐
│                STAGE 2: FACE IDENTITY DETAILING                 │
└─────────────────────────────────────────────────────────────────┘

Input: Base Image + Face ID Images
         │                    │
         ▼                    ▼
┌─────────────────┐    ┌─────────────────┐
│ Face Detection  │    │Face Analysis    │
│ InsightFace     │    │PhotoMaker V2    │
│ Landmark Detect │    │Identity Extract │
└─────┬───────────┘    └─────┬───────────┘
      │                      │
      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐
│ Mask Generation │    │Face Conditioning│
│Progressive Masks│    │[2,77+N,2048]    │
│[T,1,H//8,W//8]  │    │ Token Injection │
└─────┬───────────┘    └─────┬───────────┘
      │                      │
      └──────────┬───────────┘
                 ▼
      ┌─────────────────┐    ┌─────────────────┐
      │Differential     │───▶│ Face Replace    │
      │Diffusion        │    │ Composite Back  │
      │20 steps, CFG=4  │    │ Final Image     │
      └─────────────────┘    └─────────────────┘
```

---

## Custom Scheduler Implementation

```python
class NoiseScheduler:
    """Production-optimized timestep scheduling for face generation."""
    
    def __init__(self, scheduler: SchedulerMixin):
        self.scheduler = scheduler
        # Empirically optimized anchor points for face generation quality
        self._default_timesteps = [999, 845, 730, 587, 443, 310, 193, 116, 53, 13]
    
    def set_timesteps(self, num_steps: int, device: torch.device) -> None:
        """Set custom timesteps using log-linear interpolation."""
        # Replace HuggingFace standard linear spacing with optimized curve
        timesteps = self._loglinear_interp(num_steps)
        self.scheduler.set_timesteps(
            num_inference_steps=None, 
            timesteps=timesteps, 
            device=device
        )
    
    def _loglinear_interp(self, num_steps: int) -> torch.Tensor:
        """Log-linear interpolation between optimized anchor points."""
        if num_steps == len(self._default_timesteps):
            return torch.tensor(self._default_timesteps[::-1])
        
        # Logarithmic interpolation for smooth quality curve
        x = np.linspace(0, 1, len(self._default_timesteps))
        log_timesteps = np.log(np.array(self._default_timesteps) + 1)
        
        x_new = np.linspace(0, 1, num_steps)
        log_interp = np.interp(x_new, x, log_timesteps)
        timesteps = np.exp(log_interp) - 1
        
        return torch.tensor(timesteps[::-1].astype(int))

# Usage in production pipeline
def generate_with_custom_scheduling(prompt: str, num_steps: int = 20):
    """Production generation with optimized timestep distribution."""
    scheduler = NoiseScheduler(DDIMScheduler.from_pretrained(...))
    scheduler.set_timesteps(num_steps, device)
    
    # Standard diffusion loop with custom timesteps
    for i, timestep in enumerate(scheduler.timesteps):
        # Custom timestep curve provides better quality/speed tradeoff
        latents = diffusion_step(latents, timestep, conditioning)
    
    return vae_decode(latents)
```

---

## Advanced Tensor Operations

```python
class BatchCFGProcessor:
    """Optimized CFG processing for batch generation."""
    
    def align_conditioning_for_batch(
        self, 
        text_conditioning: torch.Tensor,  # [2*N, 77, 2048]
        spatial_ids: torch.Tensor,       # [2, 6] 
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Align conditioning tensors for efficient batch CFG processing.
        
        Standard pattern: [neg1, pos1, neg2, pos2] - per-image interleaving
        Optimized pattern: [neg1, neg2, pos1, pos2] - type-grouped for efficiency
        """
        # Reshape from interleaved to grouped pattern
        conditioning_reshaped = text_conditioning.view(2, batch_size, 77, 2048)
        negative_cond = conditioning_reshaped[0]  # [N, 77, 2048]
        positive_cond = conditioning_reshaped[1]  # [N, 77, 2048]
        
        # Group by type for tensor efficiency
        aligned_conditioning = torch.cat([negative_cond, positive_cond], dim=0)
        
        # Expand spatial IDs for batch processing
        spatial_ids_expanded = spatial_ids.repeat(batch_size, 1)  # [2*N, 6]
        
        return aligned_conditioning, spatial_ids_expanded


class ProgressiveMaskProcessor:
    """Progressive masking for differential diffusion."""
    
    def generate_progressive_masks(
        self,
        face_mask: torch.Tensor,  # [N, 1, H//8, W//8]
        timesteps: torch.Tensor,  # [T]
    ) -> torch.Tensor:
        """Generate progressive masks for differential diffusion.
        
        Creates masks that gradually expose face regions across timesteps,
        enabling precise control over face identity application.
        """
        num_steps = len(timesteps)
        batch_size = face_mask.shape[0]
        
        # Create threshold progression from 0 to 1
        thresholds = torch.linspace(0.0, 1.0, num_steps)  # [T]
        
        # Broadcast for tensor comparison: [T, 1, 1, 1] vs [1, N, H//8, W//8]
        thresholds_expanded = thresholds[:, None, None, None]  # [T, 1, 1, 1]
        mask_expanded = face_mask[None, :, :, :]  # [1, N, 1, H//8, W//8]
        
        # Progressive mask: early timesteps = small mask, later = full mask
        progressive_masks = (thresholds_expanded < mask_expanded).float()
        
        return progressive_masks  # [T, N, 1, H//8, W//8]
    
    def apply_differential_diffusion(
        self,
        predicted_noise: torch.Tensor,   # [2*N, 4, H//8, W//8] (CFG expanded)
        progressive_masks: torch.Tensor, # [T, N, 1, H//8, W//8]
        timestep_idx: int
    ) -> torch.Tensor:
        """Apply progressive masking to noise prediction."""
        current_mask = progressive_masks[timestep_idx]  # [N, 1, H//8, W//8]
        
        # Expand mask for CFG (negative + positive)
        cfg_mask = current_mask.repeat(2, 1, 1, 1)  # [2*N, 1, H//8, W//8]
        
        # Expand to match noise channels
        channel_mask = cfg_mask.repeat(1, 4, 1, 1)  # [2*N, 4, H//8, W//8]
        
        # Apply mask: only process noise within face regions
        masked_noise = predicted_noise * channel_mask
        
        return masked_noise
```

---

## Infrastructure as Code Example

```hcl
# Multi-tier GPU capacity with cost optimization
resource "aws_ecs_capacity_provider" "g6" {
  name = "g6-primary"
  
  auto_scaling_group_provider {
    auto_scaling_group_arn = aws_autoscaling_group.g6.arn
    
    managed_scaling {
      status = "DISABLED"  # Manual control for cost optimization
      target_capacity = 100
      minimum_scaling_step_size = 1
      maximum_scaling_step_size = 1
    }
  }
}

resource "aws_launch_template" "g6_template" {
  name_prefix   = "ecs-g6-"
  image_id      = data.aws_ssm_parameter.gpu_ecs_optimized_ami.value
  instance_type = "g6.xlarge"  # Latest Ada Lovelace architecture
  
  # On-demand for reliability (commented spot for primary)
  # instance_market_options {
  #   market_type = "spot"
  # }
  
  user_data = base64encode(templatefile("${path.module}/gpu_userdata.sh", {
    models_bucket = local.models_bucket_to_mount.name
    app_bucket    = local.app_bucket_to_mount.name
    cache_path    = local.cache_models_path
  }))
  
  block_device_mappings {
    device_name = "/dev/xvda"
    ebs {
      volume_type           = "gp3"
      volume_size           = 50
      encrypted             = false
      delete_on_termination = true
    }
  }
}

# Generation service with GPU resource requirements
module "generation_service" {
  source = "../../../modules/ecs_service"
  
  name      = "generation"
  cluster_id = aws_ecs_cluster.main.id
  
  # Resource allocation for ML workload
  cpu               = 4 * 1024      # 4 vCPU for model inference parallelism
  hard_memory_limit = floor(0.85 * 16 * 1024)  # 85% of 16GB for model loading
  
  resource_requirements = [
    { type = "GPU", value = "1" }  # Single GPU requirement
  ]
  
  # Multi-tier capacity strategy
  capacity_provider_strategy = [
    {
      capacity_provider = aws_ecs_capacity_provider.g6.name
      weight           = 3  # Prefer G6 on-demand
      base             = 1
    },
    {
      capacity_provider = aws_ecs_capacity_provider.g5.name  
      weight           = 2  # Fallback to G5 spot
      base             = 0
    },
    {
      capacity_provider = aws_ecs_capacity_provider.g4.name
      weight           = 1  # Final fallback G4 spot
      base             = 0
    }
  ]
  
  # S3 mount points for model storage
  mount_points = [
    { containerPath = "/models", sourceVolume = "models" },
    { containerPath = "/data", sourceVolume = "app-data" },
    { containerPath = "/cache_models", sourceVolume = "cache_models" }
  ]
  
  # Production deployment settings
  deployment_healthy_requirements = {
    minimum_percent = 0    # Allow full replacement during deployment
    maximum_percent = 200  # Enable blue-green deployment pattern
  }
}
```

---

## PhotoMaker Identity Preservation & Differential Diffusion

### PhotoMaker ID Encoder Architecture

The system implements a sophisticated face identity preservation mechanism using PhotoMaker V2 with custom tensor processing for precise face identity injection.

```python
class IDEncoder(CLIPVisionModelWithProjection):
    """PhotoMaker identity encoder with custom face processing components."""
    
    def __init__(self, id_embeddings_dim=512):
        super().__init__(CLIPVisionConfig(**VISION_CONFIG_DICT))
        
        # Core PhotoMaker components
        self.fuse_module = FuseModule(2048)  # Face-text fusion
        self.visual_projection_2 = nn.Linear(1024, 1280, bias=False)
        
        # Face token processing
        self.num_tokens = 2  # PhotoMaker uses 2 tokens per face
        self.cross_attention_dim = 2048
        self.qformer_perceiver = QFormerPerceiver(
            id_embeddings_dim, 
            self.cross_attention_dim, 
            self.num_tokens,
        )

    def forward(self, id_pixel_values, prompt_embeds, class_tokens_mask, id_embeds):
        """Process face images and inject identity into text embeddings.
        
        Tensor Flow:
        1. id_pixel_values: [b, num_inputs, c, h, w] → CLIP vision processing
        2. id_embeds: [b*num_inputs, -1] → QFormer perceiver processing  
        3. Face tokens: [b, num_inputs, 2, 2048] → Identity representation
        4. Fused embeddings: [b, 77, 2048] → Text + face identity
        """
        b, num_inputs, c, h, w = id_pixel_values.shape
        id_pixel_values = id_pixel_values.view(b * num_inputs, c, h, w)
        
        # Extract visual features using CLIP vision backbone
        last_hidden_state = self.vision_model(id_pixel_values)[0]
        id_embeds = id_embeds.view(b * num_inputs, -1)
        
        # Process through QFormer perceiver for face token generation
        id_embeds = self.qformer_perceiver(id_embeds, last_hidden_state)
        id_embeds = id_embeds.view(b, num_inputs, self.num_tokens, -1)
        
        # Fuse face tokens with text embeddings at trigger word positions
        updated_prompt_embeds = self.fuse_module(prompt_embeds, id_embeds, class_tokens_mask)
        
        return updated_prompt_embeds


class FuseModule(nn.Module):
    """Fuses face identity tokens with text embeddings at specific positions."""
    
    def __init__(self, embed_dim):
        super().__init__()
        self.mlp1 = MLP(embed_dim * 2, embed_dim, embed_dim, use_residual=False)
        self.mlp2 = MLP(embed_dim, embed_dim, embed_dim, use_residual=True)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def fuse_fn(self, prompt_embeds, id_embeds):
        """Core fusion operation: combine text and face embeddings."""
        stacked_id_embeds = torch.cat([prompt_embeds, id_embeds], dim=-1)
        stacked_id_embeds = self.mlp1(stacked_id_embeds) + prompt_embeds
        stacked_id_embeds = self.mlp2(stacked_id_embeds)
        stacked_id_embeds = self.layer_norm(stacked_id_embeds)
        return stacked_id_embeds

    def forward(self, prompt_embeds, id_embeds, class_tokens_mask):
        """Replace trigger word embeddings with fused face+text tokens.
        
        Critical Operation:
        - prompt_embeds: [b, 77, 2048] - Standard CLIP text embeddings
        - id_embeds: [b, num_faces, 2, 2048] - Face identity tokens
        - class_tokens_mask: [b, 77] - Boolean mask for trigger word positions
        
        Result: Text embeddings with face identity injected at "img" token positions
        """
        id_embeds = id_embeds.to(prompt_embeds.dtype)
        num_inputs = class_tokens_mask.sum().unsqueeze(0)
        batch_size, max_num_inputs = id_embeds.shape[:2]
        seq_length = prompt_embeds.shape[1]
        
        # Flatten and filter valid face embeddings
        flat_id_embeds = id_embeds.view(-1, id_embeds.shape[-2], id_embeds.shape[-1])
        valid_id_mask = (
            torch.arange(max_num_inputs, device=flat_id_embeds.device)[None, :]
            < num_inputs[:, None]
        )
        valid_id_embeds = flat_id_embeds[valid_id_mask.flatten()]
        
        # Process embeddings and mask for token replacement
        prompt_embeds = prompt_embeds.view(-1, prompt_embeds.shape[-1])
        class_tokens_mask = class_tokens_mask.view(-1)
        valid_id_embeds = valid_id_embeds.view(-1, valid_id_embeds.shape[-1])
        
        # Extract and fuse trigger word embeddings with face tokens
        image_token_embeds = prompt_embeds[class_tokens_mask]
        stacked_id_embeds = self.fuse_fn(image_token_embeds, valid_id_embeds)
        
        # Replace trigger word positions with fused embeddings
        prompt_embeds.masked_scatter_(class_tokens_mask[:, None], stacked_id_embeds.to(prompt_embeds.dtype))
        updated_prompt_embeds = prompt_embeds.view(batch_size, seq_length, -1)
        
        return updated_prompt_embeds
```

### Face Identity Processing Pipeline

```python
class ConditioningService:
    """Prepares text and face embeddings with PhotoMaker integration."""
    
    def prepare_conditioning(self, params: ConditioningParams) -> Conditioning:
        """Prepare conditioning with face identity injection.
        
        Process Flow:
        1. Extract face embeddings using InsightFace
        2. Augment prompt with gender detection + trigger word
        3. Create text embeddings with and without face conditioning
        4. Generate spatial conditioning for SDXL
        """
        prompt = params.prompt
        negative_prompt = params.negative_prompt or ""
        face_images = params.face_images
        face_id_weight = params.face_id_weight
        
        # Expand face images to match face_id_weight (strength parameter)
        expanded_face_images = []
        for i in range(face_id_weight):
            expanded_face_images.append(face_images[i % len(face_images)])
        
        # Extract face embeddings and gender detection
        face_embeddings = self.face_embedder.compute_embeddings(expanded_face_images)
        
        # Augment prompt with detected gender and trigger word
        augmented_prompt = f"{face_embeddings.gender} {self.trigger_word} {prompt}"
        
        # Create text-only conditioning (for base generation)
        prompt_without_trigger = augmented_prompt.replace(self.trigger_word, "").strip()
        detailed_text, global_text = self.text_encoder.encode(prompt_without_trigger, self.device)
        detailed_neg, global_neg = self.text_encoder.encode(negative_prompt, self.device)
        
        # Create face conditioning through PhotoMaker ID encoder
        face_conditioning = self._prepare_face_conditioning(
            expanded_face_images, face_embeddings, augmented_prompt, negative_prompt
        )
        
        # Stack negative + positive for CFG
        detailed_text_combined = torch.cat([detailed_neg, detailed_text], dim=0)
        global_text_combined = torch.cat([global_neg, global_text], dim=0)
        
        return Conditioning(
            detailed_text=detailed_text_combined,
            global_text=global_text_combined,
            detailed_text_face=face_conditioning.detailed,
            global_text_face=face_conditioning.global_pooled,
            spatial_ids=params.spatial_ids
        )
    
    def _prepare_face_conditioning(self, face_images, face_embeddings, prompt, negative_prompt):
        """Process face images through PhotoMaker ID encoder."""
        # Process face images through CLIP image processor
        face_pixel_values = self.id_image_processor(
            face_images, return_tensors="pt"
        ).pixel_values.to(self.device, dtype=self.dtype)
        
        # Add batch dimension: [num_faces, c, h, w] → [1, num_faces, c, h, w]
        face_pixel_values = face_pixel_values.unsqueeze(0)
        
        # Create text embeddings with trigger word
        detailed_pos, global_pos = self.text_encoder.encode(prompt, self.device)
        detailed_neg, global_neg = self.text_encoder.encode(negative_prompt, self.device)
        
        # Find trigger word positions for face token injection
        class_tokens_mask = self._find_trigger_positions(prompt)
        
        # Convert face embeddings to tensor format
        face_embeds = torch.tensor(
            face_embeddings.embeddings, device=self.device, dtype=self.dtype
        ).unsqueeze(0)  # [1, num_faces, embedding_dim]
        
        # Process positive conditioning through ID encoder
        updated_detailed_pos = self.id_encoder(
            face_pixel_values, detailed_pos, class_tokens_mask, face_embeds
        )
        
        # Process negative conditioning (no face injection)
        updated_detailed_neg = detailed_neg  # No face conditioning for negative
        
        # Combine for CFG
        detailed_combined = torch.cat([updated_detailed_neg, updated_detailed_pos], dim=0)
        global_combined = torch.cat([global_neg, global_pos], dim=0)
        
        return FaceConditioning(
            detailed=detailed_combined,
            global_pooled=global_combined
        )
```

### Differential Diffusion Implementation

```python
def _batch_differential_diffusion(
    self,
    magnified_faces: torch.Tensor,  # [N_faces, 3, 1024, 1024]
    mask_init: torch.Tensor,        # [N_faces, 1, 1024, 1024]
    conditioning: Conditioning,
    params: DetailingParams,
    seeds: List[torch.Generator],
    target_resolution: int
) -> torch.Tensor:
    """Perform differential diffusion with progressive masking.
    
    Critical Tensor Transformations:
    1. VAE Encoding: [N, 3, 1024, 1024] → [N, 4, 128, 128] (latent space)
    2. Mask Processing: [N, 1, 1024, 1024] → [N, 1, 128, 128] → [steps, N, 128, 128]
    3. Progressive Masking: Gradual face region exposure across timesteps
    4. CFG Expansion: [N, 4, 128, 128] → [2*N, 4, 128, 128]
    5. UNet Processing: Noise prediction with face conditioning
    6. VAE Decoding: [N, 4, 128, 128] → [N, 3, 1024, 1024]
    """
    
    # Setup denoising parameters
    steps = params.steps
    original_steps = steps  # Critical: preserve for mask creation
    denoise_strength = params.denoise
    guidance_scale = params.guidance_scale
    
    # Set custom timestep schedule
    self.scheduler.set_timesteps(steps, self.device)
    timesteps = self.scheduler.timesteps
    
    # Calculate denoising range
    init_timestep = min(int(steps * denoise_strength), steps)
    t_start = max(steps - init_timestep, 0)
    timesteps = timesteps[t_start:]
    
    # Encode faces to latent space
    init_latents = self.vae_processor.encode(magnified_faces * 2 - 1, seeds)
    
    # Create noise for denoising
    noise = torch.randn(
        init_latents.shape, generator=seeds[0], device=self.device, dtype=torch.float16
    ).unsqueeze(0)
    
    # Add noise according to timestep schedule
    original_noise_desc = self.scheduler.add_noise(
        init_latents.unsqueeze(0), noise, timesteps
    )
    
    # Process masks for latent space (CRITICAL: preserve 4D for interpolation)
    vae_scale_factor = 8
    latent_resolution = target_resolution // vae_scale_factor
    
    # Resize mask to latent dimensions
    mask_resized = torch.nn.functional.interpolate(
        mask_init,  # [N_faces, 1, 1024, 1024]
        size=(latent_resolution, latent_resolution),  # → [N_faces, 1, 128, 128]
        mode='bilinear',
        align_corners=False,
        antialias=None
    )
    
    # Create progressive masks using tensor broadcasting
    total_time_steps = original_steps
    thresholds = torch.arange(total_time_steps, dtype=mask_resized.dtype) / total_time_steps
    thresholds = thresholds.to(self.device)  # [timesteps]
    
    # Remove channel dimension for broadcasting
    mask_no_channel = mask_resized.squeeze(1)  # [N_faces, 128, 128]
    
    # CRITICAL: Progressive mask creation via broadcasting
    # thresholds: [timesteps] → [timesteps, 1, 1, 1]  
    # mask_no_channel: [N_faces, 128, 128] → [1, N_faces, 128, 128]
    # Result: [timesteps, N_faces, 128, 128]
    progressive_masks = thresholds[:, None, None, None] < mask_no_channel[None, :, :, :]
    
    # Denoising loop with progressive masking
    latents = original_noise_desc[0]  # Start from first timestep
    
    for i, timestep in enumerate(timesteps):
        # Get current progressive mask
        current_mask = progressive_masks[i]  # [N_faces, 128, 128]
        
        # CFG: duplicate latents for conditional/unconditional processing
        latent_model_input = torch.cat([latents] * 2, dim=0)  # [2*N_faces, 4, 128, 128]
        
        # Scale model input according to scheduler
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep)
        
        # UNet forward pass with face conditioning
        noise_pred = self.unet.forward(
            sample=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=conditioning.detailed_text_face,  # Face conditioning
            added_cond_kwargs={
                "text_embeds": conditioning.global_text_face,
                "time_ids": conditioning.spatial_ids
            }
        )
        
        # CFG guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # Apply progressive mask to noise prediction (differential diffusion)
        mask_expanded = current_mask.unsqueeze(1).repeat(1, 4, 1, 1)  # [N_faces, 4, 128, 128]
        noise_pred = noise_pred * mask_expanded
        
        # Scheduler step
        latents = self.scheduler.step(noise_pred, timestep, latents).prev_sample
    
    # Decode refined latents back to image space
    refined_faces = self.vae_processor.decode(latents)  # [N_faces, 3, 1024, 1024]
    
    return refined_faces
```

### Model Registry & LoRA Integration

```python
class ModelRegistry:
    """Manages PhotoMaker LoRA integration with memory optimization."""
    
    def load_photomaker(self, photomaker_path: str, insightface_path: str):
        """Load PhotoMaker with careful memory management."""
        
        # Load PhotoMaker state dict
        state_dict = torch.load(photomaker_path, map_location="cpu")
        
        # Load ID encoder component
        from .id_encoder import IDEncoder
        id_encoder = IDEncoder()
        id_encoder.load_state_dict(state_dict["id_encoder"], strict=True)
        id_encoder = id_encoder.to(self.device, dtype=self.dtype)
        
        # CRITICAL: Load LoRA weights using existing pipeline
        logger.info("Loading PhotoMaker LoRA weights")
        self._base_pipeline.load_lora_weights(
            state_dict["lora_weights"], 
            adapter_name="photomaker"
        )
        self._base_pipeline.fuse_lora()
        
        # Memory optimization: delete pipeline after LoRA fusion
        del self._base_pipeline
        self._base_pipeline = None
        torch.cuda.empty_cache()
        
        # Add trigger word to tokenizers
        self.trigger_word = "img"
        self.tokenizer.add_tokens([self.trigger_word], special_tokens=True)
        self.tokenizer_2.add_tokens([self.trigger_word], special_tokens=True)
        
        # Initialize face analysis
        self.face_embedder = FaceEmbedder(
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            allowed_modules=["detection", "recognition", "genderage", "landmark_2d_106"],
            root=insightface_path
        )
        self.face_embedder.prepare(ctx_id=0, det_size=(640, 640))
        
        # Cache components
        self.id_encoder = id_encoder
        self.id_image_processor = CLIPImageProcessor()
```

This implementation showcases the sophisticated face identity preservation mechanism combining PhotoMaker V2, custom tensor operations, and differential diffusion for precise face identity injection while maintaining high generation quality.

---

## Part 3: Production Operations
*Supporting Page 3: Production Operations & Web Platform*

These examples demonstrate production operations patterns that show how infrastructure expertise enables real-world AI deployment. The Discord bot commands, web platform architecture, and caching strategies shown here apply the same operational practices used in distributed systems to ML production environments.

### Discord Bot Command Interface

```python
# Unified parameter management system (from command_service.py)
class CommandService(commands.Cog):
    """Production Discord operations interface with parameter validation."""
    
    PARAMETER_DEFINITIONS = {
        "guidance_scale": {"type": float, "min": 1.0, "max": 30.0},
        "steps": {"type": int, "min": 1, "max": 150}, 
        "weight": {"type": float, "min": 0.0, "max": 2.0},
        "production_mode": {"type": bool}
    }

    @commands.slash_command(description="Set generation parameters")
    async def set(self, inter, parameter: str, value: str):
        """Set individual parameters with validation and autocomplete."""
        
    @commands.slash_command(description="Test faces with Mecene-enhanced inputs")
    async def test_face_mecene(self, inter, face_folders: str, generations_per_input: int = 3):
        """Execute batch generations: face_folders × mecene_ideas × generations_per_input."""
        
    @commands.slash_command(description="Run parameter sweep using YAML configuration")
    async def sweep(self, inter, config: str):
        """Execute parameter combinations for systematic testing."""
```

### PhotoWall Global Cache System

```typescript
// Unified cache with metadata and image buffers (from photowall.server.ts)
declare global {
    var photoWallCache: {
        celebrityPhotos: Map<string, Photo[]>
        availableCelebrities: Celebrity[]
        lastUpdated: number
        imageStats: {
            totalImages: number
            totalSizeBytes: number
            imagesLoaded: number
        }
    }
}

// Global cache loading with emergency reload capability
export async function loadPhotowallData(): Promise<boolean> {
    try {
        // Load celebrity index and filter by whitelist
        const index: CelebrityIndex = JSON.parse(fs.readFileSync(indexPath, 'utf-8'))
        global.photoWallCache.availableCelebrities = index.celebrities
        
        // Preload all images into unified cache (metadata + buffers)
        for (const celebrity of global.photoWallCache.availableCelebrities) {
            const photos = getCelebrityPhotos(celebrity.id)
            global.photoWallCache.celebrityPhotos.set(celebrity.id, photos)
        }
        
        return global.photoWallCache.celebrityPhotos.size > 0
    } catch (error) {
        logger.error('Cache loading failed:', error)
        return false
    }
}
```

### PostgreSQL Session Management

```typescript
// Production session management with token transfer (from session.repository.ts)
export class PostgresSessionRepository implements SessionRepository {
    async linkUser(sessionId: string, userId: string): Promise<void> {
        // Infrastructure pattern: atomic multi-table operations
        const session = await this.findById(sessionId)
        const user = await prisma.user.findUnique({ where: { id: userId } })
        const totalTokens = session.tokens + user.tokens
        
        // Atomic token transfer operation
        await Promise.all([
            this.update(sessionId, { userId, tokens: 0 }),
            prisma.user.update({ where: { id: userId }, data: { tokens: totalTokens } })
        ])
    }
}
```

---

These technical materials demonstrate the unified nature of infrastructure and ML systems engineering. The same patterns that ensure reliability in distributed systems—parameter validation, global caching, atomic operations, graceful degradation—directly enable production AI systems that are reliable, observable, and operationally manageable.