import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_seq_length=1000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        
        # Precomputing positional encodings
        pe = torch.zeros(max_seq_length, embedding_dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * 
                           (-np.log(10000.0) / embedding_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_seq_length, embedding_dim]
        
    def forward(self, x):
        """
        Add positional encoding to input features.
        Args: x: [batch_size, seq_len, embedding_dim]  
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class HeatmapEncoder(nn.Module):
    """
    Enhanced encoder with noise reduction capabilities
    """
    def __init__(self, heatmap_size=(288, 512), embed_dim=256):
        super().__init__()
        self.heatmap_size = heatmap_size
        self.embed_dim = embed_dim
        self.flatten_dim = heatmap_size[0] * heatmap_size[1]
        
        # Add initial denoising layer
        self.denoise_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # Linear projection from flattened heatmap to embedding
        self.projection = nn.Sequential(
            nn.Linear(self.flatten_dim, embed_dim * 2),
            nn.BatchNorm1d(embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.05),  # Reduced dropout
            nn.Linear(embed_dim * 2, embed_dim),
            nn.BatchNorm1d(embed_dim)
        )
        
    def forward(self, heatmaps):
        """
        Args: heatmaps: [batch_size, seq_len, height, width]
        Returns: features: [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, height, width = heatmaps.shape
        
        # Reshape for conv2d processing
        heatmaps_reshaped = heatmaps.view(batch_size * seq_len, 1, height, width)
        
        # Apply denoising convolution
        denoised = self.denoise_conv(heatmaps_reshaped)
        
        # Reshape back and flatten
        denoised = denoised.view(batch_size, seq_len, height, width)
        flattened = denoised.view(batch_size, seq_len, -1)
        
        # Project to embedding space
        features = self.projection(flattened.view(-1, self.flatten_dim))
        features = features.view(batch_size, seq_len, self.embed_dim)
        
        return features


class HeatmapDecoder(nn.Module):
    """
    Enhanced decoder with noise suppression
    """
    def __init__(self, heatmap_size=(288, 512), embed_dim=256):
        super().__init__()
        self.heatmap_size = heatmap_size
        self.embed_dim = embed_dim
        self.flatten_dim = heatmap_size[0] * heatmap_size[1]
        
        # Linear projection from embedding to flattened heatmap
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.BatchNorm1d(embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.05),  # Reduced dropout
            nn.Linear(embed_dim * 2, embed_dim * 4),
            nn.BatchNorm1d(embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.05),
            nn.Linear(embed_dim * 4, self.flatten_dim)
        )
        
        self.denoise_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # Gaussian smoothing kernel for final cleanup
        self.register_buffer('gaussian_kernel', self._create_gaussian_kernel(5, 1.0))
        
    def _create_gaussian_kernel(self, size, sigma):
        """Create a Gaussian kernel for smoothing"""
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        kernel = g.outer(g).unsqueeze(0).unsqueeze(0)
        return kernel
        
    def forward(self, features, original_heatmaps=None):
        """
        Args: 
            features: [batch_size, seq_len, embed_dim]
            original_heatmaps: [batch_size, seq_len, height, width] (optional skip connection)
        Returns: heatmaps: [batch_size, seq_len, height, width]
        """
        batch_size, seq_len, embed_dim = features.shape
        
        # Project to heatmap space
        flattened_heatmaps = self.projection(features.view(-1, embed_dim))
        
        heatmaps = flattened_heatmaps.view(batch_size, seq_len, 
                                         self.heatmap_size[0], self.heatmap_size[1])
        
        heatmaps = torch.sigmoid(heatmaps)
        
        # Reshape for conv processing
        heatmaps_reshaped = heatmaps.view(batch_size * seq_len, 1, 
                                        self.heatmap_size[0], self.heatmap_size[1])
        
        # Apply denoising convolution
        denoised = self.denoise_conv(heatmaps_reshaped)
        
        # Apply Gaussian smoothing for final cleanup
        denoised = F.conv2d(denoised, self.gaussian_kernel, padding=2)
        
        # Reshape back
        denoised = denoised.view(batch_size, seq_len, 
                               self.heatmap_size[0], self.heatmap_size[1])
        
        # Optional skip connection with original heatmaps (reduced weight)
        if original_heatmaps is not None:
            # More conservative skip connection
            denoised = 0.8 * denoised + 0.2 * original_heatmaps
        
        return denoised


class TemporalTransformer(nn.Module):
    """
    Simplified temporal transformer with reduced complexity to minimize noise
    """
    def __init__(self, embed_dim=256, num_heads=8, num_layers=3, dropout=0.02):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Simplified transformer layers
        self.temporal_layers = nn.ModuleList()
        for i in range(num_layers):
            self.temporal_layers.append(self._build_temporal_layer(embed_dim, num_heads, dropout))
        
        self.final_norm = nn.LayerNorm(embed_dim)
        
    def _build_temporal_layer(self, embed_dim, num_heads, dropout):
        """Build a simpler temporal transformer layer"""
        return nn.ModuleDict({
            'attention': nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True),
            'feedforward': nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Dropout(dropout * 0.5)
            ),
            'norm1': nn.LayerNorm(embed_dim),
            'norm2': nn.LayerNorm(embed_dim)
        })
    
    def forward(self, features):
        x = features
        
        for layer in self.temporal_layers:
            # Self-attention with residual connection
            attn_out, _ = layer['attention'](x, x, x)
            x = layer['norm1'](x + attn_out)
            
            # Feedforward with residual connection
            ff_out = layer['feedforward'](x)
            x = layer['norm2'](x + ff_out)
        
        return self.final_norm(x)


class Transformer(nn.Module):
    """
    Improved balltracking transformer with noise reduction
    """
    def __init__(self, num_heads=8, num_layers=3, heatmap_size=(288, 512), embed_dim=256):
        super().__init__()
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.heatmap_size = heatmap_size
        self.embed_dim = embed_dim
        
        self.encoder = HeatmapEncoder(heatmap_size, embed_dim)
        self.decoder = HeatmapDecoder(heatmap_size, embed_dim)
        self.temporal_transformer = TemporalTransformer(embed_dim, num_heads, num_layers)
        self.positional_encoding = PositionalEncoding(embed_dim)

    def forward(self, heatmaps):
        """
        Main forward pass for ball tracking.
        Args: heatmaps: [batch_size, seq_len, height, width]
        Returns: refined_heatmaps: [batch_size, seq_len, height, width]
        """
        # Encoding heatmaps to embedding space
        features = self.encoder(heatmaps)
        
        features_with_pos = self.positional_encoding(features)
        temporal_features = self.temporal_transformer(features_with_pos)
        refined_heatmaps = self.decoder(temporal_features, heatmaps)
        
        return refined_heatmaps
    
    def extract_detections(self, heatmaps, threshold=0.3):  # Increased threshold
        """
        Extract 2D pixel detections from heatmaps using argmax with confidence filtering.
        Args: 
            heatmaps: [batch_size, seq_len, height, width]
            threshold: minimum confidence threshold for valid detections
        Returns: 
            detections: [batch_size, seq_len, 2] (x, y coordinates, -1 for invalid)
            confidences: [batch_size, seq_len] confidence values
        """
        batch_size, seq_len, height, width = heatmaps.shape
        
        # Apply additional smoothing before detection
        smoothed_heatmaps = F.avg_pool2d(
            heatmaps.view(-1, 1, height, width), 
            kernel_size=3, stride=1, padding=1
        ).view(batch_size, seq_len, height, width)
        
        # Flatten spatial dimensions
        flattened = smoothed_heatmaps.view(batch_size, seq_len, -1)
        
        # Get max values and indices
        max_values, max_indices = torch.max(flattened, dim=2)
        
        # Create confidence mask
        mask = max_values > threshold
        
        # Convert indices to coordinates
        y_coords = (max_indices // width).float()
        x_coords = (max_indices % width).float()
        
        # Stack coordinates
        detections = torch.stack([x_coords, y_coords], dim=2)
        
        # Mark invalid detections
        detections[~mask] = -1
        
        return detections, max_values
    
    def get_debug_info(self, heatmaps):
        """
        Get intermediate outputs for debugging
        """
        # Encode heatmaps
        features = self.encoder(heatmaps)
        
        # Add positional encoding
        features_with_pos = self.positional_encoding(features)
        temporal_features = self.temporal_transformer(features_with_pos)
        decoded_without_skip = self.decoder(temporal_features, None)
        
        refined_heatmaps = self.decoder(temporal_features, heatmaps)
        
        return {
            'encoded_features': features,
            'features_with_pos': features_with_pos,
            'temporal_features': temporal_features,
            'decoded_without_skip': decoded_without_skip,
            'refined_heatmaps': refined_heatmaps,
            'input_stats': {
                'mean': heatmaps.mean().item(),
                'std': heatmaps.std().item(),
                'max': heatmaps.max().item(),
                'min': heatmaps.min().item()
            }
        }
