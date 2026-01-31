#!/usr/bin/env python3
"""
Acoustic Material to ITU-R P.2040 Parameter Converter

This script converts acoustic material properties to ITU-R P.2040 format
for use in radio wave ray-tracing tools to simulate acoustic propagation.

The key insight is that acoustic waves at kHz frequencies have similar
wavelengths to radio waves at GHz frequencies:
    λ_acoustic(1 kHz) ≈ λ_radio(875 MHz) ≈ 0.34 m

Frequency scaling factor: f_radio = f_acoustic × (c_radio / c_sound)
    = f_acoustic × (3×10^8 / 343) ≈ f_acoustic × 875,000

ITU-R P.2040 Model:
    ε'(f) = a × f^b      (f in GHz, ε' dimensionless)
    σ(f)  = c × f^d      (f in GHz, σ in S/m)
    ε''   = 17.98 × σ / f_GHz

Author: Acoustic-to-RF Mapping Framework
Date: 2025
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List
from enum import Enum
import json


# =============================================================================
# Physical Constants
# =============================================================================

C_SOUND = 343.0          # Speed of sound in air at 20°C (m/s)
C_LIGHT = 3e8            # Speed of light (m/s)
RHO_AIR = 1.2            # Air density (kg/m³)
Z_AIR = RHO_AIR * C_SOUND  # Acoustic impedance of air ≈ 413 rayls

# Frequency scaling factor: converts acoustic Hz to radio GHz
# f_radio_GHz = f_acoustic_Hz × FREQ_SCALE_TO_GHZ
FREQ_SCALE_TO_GHZ = (C_LIGHT / C_SOUND) / 1e9  # ≈ 8.75×10^-4

# Permittivity of free space
EPSILON_0 = 8.854e-12    # F/m


# =============================================================================
# Material Classes
# =============================================================================

class MaterialType(Enum):
    """Classification of acoustic materials by their wave interaction behavior."""
    RIGID = "rigid"           # Hard surfaces: concrete, glass, steel
    POROUS = "porous"         # Fibrous/foam absorbers
    EMPIRICAL = "empirical"   # Furniture, people - use measured α coefficients


@dataclass
class AcousticMaterial:
    """
    Acoustic material properties.
    
    Attributes:
        name: Material identifier
        material_type: Classification (RIGID, POROUS, or EMPIRICAL)
        
        # For RIGID materials:
        rho: Density (kg/m³)
        c: Speed of sound in material (m/s)
        
        # For POROUS materials:
        flow_resistivity: Flow resistivity σ_f (rayls/m or Pa·s/m²)
        thickness: Typical thickness (m) - used for reference
        
        # For EMPIRICAL materials:
        alpha_coeffs: Absorption coefficients at standard octave bands
                      [125, 250, 500, 1000, 2000, 4000] Hz
        d_eff: Effective acoustic thickness (m)
    """
    name: str
    material_type: MaterialType
    
    # Rigid material parameters
    rho: Optional[float] = None           # kg/m³
    c: Optional[float] = None             # m/s
    
    # Porous material parameters  
    flow_resistivity: Optional[float] = None  # rayls/m
    thickness: Optional[float] = None         # m
    
    # Empirical material parameters
    alpha_coeffs: Optional[List[float]] = None  # α at [125,250,500,1k,2k,4k] Hz
    d_eff: Optional[float] = None               # effective thickness (m)
    
    @property
    def Z(self) -> Optional[float]:
        """Acoustic impedance Z = ρc (rayls) for rigid materials."""
        if self.rho is not None and self.c is not None:
            return self.rho * self.c
        return None


@dataclass
class P2040Parameters:
    """
    ITU-R P.2040 material parameters.
    
    Model: ε'(f) = a × f^b,  σ(f) = c × f^d  (f in GHz)
    
    Attributes:
        name: Material identifier
        a, b: Real permittivity parameters (ε' = a × f^b)
        c, d: Conductivity parameters (σ = c × f^d)
        freq_min_ghz: Minimum valid frequency (GHz)
        freq_max_ghz: Maximum valid frequency (GHz)
        notes: Additional information
    """
    name: str
    a: float
    b: float
    c: float
    d: float
    freq_min_ghz: float = 0.0262  # 30 Hz acoustic
    freq_max_ghz: float = 7.0     # 8 kHz acoustic
    notes: str = ""
    
    def epsilon_real(self, f_ghz: float) -> float:
        """Calculate real permittivity ε' at frequency f (GHz)."""
        return self.a * (f_ghz ** self.b)
    
    def sigma(self, f_ghz: float) -> float:
        """Calculate conductivity σ (S/m) at frequency f (GHz)."""
        return self.c * (f_ghz ** self.d)
    
    def epsilon_imag(self, f_ghz: float) -> float:
        """Calculate imaginary permittivity ε'' at frequency f (GHz)."""
        # ε'' = σ / (ω × ε₀) = σ / (2π × f × ε₀)
        # In practical units: ε'' = 17.98 × σ / f_GHz
        sigma = self.sigma(f_ghz)
        return 17.98 * sigma / f_ghz if f_ghz > 0 else 0.0
    
    def to_dict(self) -> dict:
        """Export parameters as dictionary."""
        return {
            "name": self.name,
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "d": self.d,
            "freq_min_ghz": self.freq_min_ghz,
            "freq_max_ghz": self.freq_max_ghz,
            "notes": self.notes
        }


# =============================================================================
# Conversion Functions
# =============================================================================

def acoustic_freq_to_radio_freq(f_acoustic_hz: float) -> float:
    """
    Convert acoustic frequency (Hz) to equivalent radio frequency (GHz).
    
    The mapping preserves wavelength: λ_acoustic ≈ λ_radio
    
    f_radio = f_acoustic × (c_light / c_sound)
    
    Example: 1000 Hz acoustic → 875 MHz radio (both have λ ≈ 0.34 m)
    """
    return f_acoustic_hz * FREQ_SCALE_TO_GHZ


def rigid_material_to_p2040(material: AcousticMaterial) -> P2040Parameters:
    """
    Convert rigid acoustic material to P.2040 parameters.
    
    For rigid materials (concrete, glass, steel, etc.):
    - Very high acoustic impedance Z >> Z_air
    - Nearly perfect reflection (|R| → 1)
    - Negligible absorption
    
    Mapping:
        ε' = (Z_air / Z_material)²
        
    This gives ε' << 1, representing the acoustic medium as "optically rarer"
    than air from the wave's perspective. The Fresnel equations then correctly
    predict near-total reflection.
    
    Since Z = ρc is frequency-independent, we have b = 0 (constant ε').
    With negligible absorption, c = d = 0.
    """
    if material.Z is None:
        raise ValueError(f"Rigid material {material.name} requires rho and c")
    
    # ε' = (Z_air / Z_material)²
    epsilon_r = (Z_AIR / material.Z) ** 2
    
    return P2040Parameters(
        name=material.name,
        a=epsilon_r,
        b=0.0,          # Frequency-independent
        c=0.0,          # No loss
        d=0.0,
        notes=f"Z = {material.Z:.2e} rayls, |R| = {abs((material.Z - Z_AIR)/(material.Z + Z_AIR)):.6f}"
    )


def delany_bazley(f_hz: float, flow_resistivity: float) -> Tuple[complex, float]:
    """
    Delany-Bazley model for porous absorber acoustic properties.
    
    This empirical model predicts the characteristic impedance and
    propagation constant of fibrous/porous materials.
    
    Args:
        f_hz: Frequency (Hz)
        flow_resistivity: σ_f (rayls/m or Pa·s/m²)
        
    Returns:
        Z_c: Complex characteristic impedance (rayls)
        alpha_m: Attenuation constant (Np/m)
        
    Reference: Delany & Bazley (1970), Applied Acoustics 3, 105-116
    """
    # Dimensionless frequency parameter
    X = RHO_AIR * f_hz / flow_resistivity
    
    # Characteristic impedance (complex)
    # Z_c / Z_air = 1 + 0.0571×X^(-0.754) - j×0.087×X^(-0.732)
    Z_c_real = Z_AIR * (1 + 0.0571 * (X ** -0.754))
    Z_c_imag = Z_AIR * (-0.087 * (X ** -0.732))
    Z_c = complex(Z_c_real, Z_c_imag)
    
    # Propagation constant
    # γ = α + jβ, where α is attenuation (Np/m)
    # α = (ω/c₀) × 0.189 × X^(-0.595)
    omega = 2 * np.pi * f_hz
    alpha_m = (omega / C_SOUND) * 0.189 * (X ** -0.595)
    
    return Z_c, alpha_m


def porous_material_to_p2040(material: AcousticMaterial, 
                              fit_freqs_hz: Optional[List[float]] = None) -> P2040Parameters:
    """
    Convert porous acoustic material to P.2040 parameters.
    
    For porous materials (foam, fiberglass, curtains, carpet):
    - Impedance closer to air (partial reflection)
    - Significant absorption due to viscous losses
    - Frequency-dependent behavior
    
    Uses Delany-Bazley model to compute Z_c(f) and α_m(f), then finds
    P.2040 parameters that reproduce the reflection behavior.
    
    Key insight: The Delany-Bazley model gives us:
    1. Z_c = complex characteristic impedance → reflection coefficient
    2. α_m = propagation attenuation → internal losses
    
    We model this as a dielectric with:
    - ε' chosen to match |R| at the surface
    - ε'' chosen to match absorption behavior
    """
    if material.flow_resistivity is None:
        raise ValueError(f"Porous material {material.name} requires flow_resistivity")
    
    sigma_f = material.flow_resistivity
    
    # Default fitting frequencies: 30 Hz to 8 kHz (acoustic)
    if fit_freqs_hz is None:
        fit_freqs_hz = [30, 63, 125, 250, 500, 1000, 2000, 4000, 8000]
    
    # Calculate properties at each frequency
    epsilon_r_values = []
    epsilon_i_values = []
    f_ghz_values = []
    
    for f_hz in fit_freqs_hz:
        f_ghz = acoustic_freq_to_radio_freq(f_hz)
        f_ghz_values.append(f_ghz)
        
        # Get Delany-Bazley complex impedance and attenuation
        Z_c, alpha_m = delany_bazley(f_hz, sigma_f)
        
        # Reflection coefficient at air-material interface
        R = (Z_c - Z_AIR) / (Z_c + Z_AIR)
        R_mag = abs(R)
        
        # For P.2040 model with ε < 1 (acoustic material "rarer" than air):
        # R = (1 - √ε) / (1 + √ε)
        # Solving for real √ε when R is real and positive:
        # √ε = (1 - R) / (1 + R)
        #
        # But R is complex. We need to match |R| for energy considerations.
        # Let's find ε' such that the reflection magnitude matches:
        # |R| = |1 - √ε| / |1 + √ε|
        #
        # For √ε = x (real, 0 < x < 1):
        # |R| = (1 - x) / (1 + x)
        # x = (1 - |R|) / (1 + |R|)
        # ε' = x²
        
        if R_mag < 0.999:
            sqrt_eps = (1 - R_mag) / (1 + R_mag)
            epsilon_r = sqrt_eps ** 2
        else:
            epsilon_r = 1e-6  # Nearly perfect reflector
        
        # For ε'', we use the propagation attenuation
        # In a dielectric with ε = ε' - jε'', the attenuation is:
        # α = (ω/c₀) × ε'' / (2√ε')  [for ε'' << ε']
        #
        # But the Delany-Bazley α_m is for the acoustic wave in the material.
        # The absorbed energy fraction for a wave entering the material is
        # related to α_m and the penetration depth.
        #
        # For our P.2040 model, we want the total absorption α = 1 - |R|²
        # to match the acoustic case. The absorption comes from:
        # 1. Reflection loss (handled by ε')
        # 2. Internal absorption (handled by ε'')
        #
        # Since our ε' already gives the correct |R|, we need ε'' to
        # provide additional losses for materials that absorb more than
        # what's predicted by just the impedance mismatch.
        
        # Acoustic absorption coefficient (total)
        alpha_total = 1 - R_mag**2
        
        # Absorption from impedance mismatch alone (our ε' model)
        R_from_eps = (1 - np.sqrt(epsilon_r)) / (1 + np.sqrt(epsilon_r))
        alpha_from_eps = 1 - R_from_eps**2
        
        # Additional absorption to add via ε''
        alpha_extra = max(0, alpha_total - alpha_from_eps)
        
        # Convert to ε'' empirically
        # Higher α_extra → higher ε''
        epsilon_i = alpha_extra * 0.3 + alpha_m * C_SOUND / (2 * np.pi * f_hz) * np.sqrt(epsilon_r)
        
        # Ensure positive values
        epsilon_r = max(epsilon_r, 1e-6)
        epsilon_i = max(epsilon_i, 1e-6)
        
        epsilon_r_values.append(epsilon_r)
        epsilon_i_values.append(epsilon_i)
    
    # Fit to power law: y = a × f^b  →  log(y) = log(a) + b×log(f)
    log_f = np.log(f_ghz_values)
    
    # Fit ε'(f) = a × f^b
    log_eps = np.log(epsilon_r_values)
    b_eps, log_a_eps = np.polyfit(log_f, log_eps, 1)
    a_eps = np.exp(log_a_eps)
    
    # Convert ε'' to σ using P.2040 formula: σ = ε'' × f_ghz / 17.98
    sigma_values = [eps_i * f / 17.98 for eps_i, f in zip(epsilon_i_values, f_ghz_values)]
    
    # Fit σ(f) = c × f^d
    log_sigma = np.log(sigma_values)
    d_sigma, log_c_sigma = np.polyfit(log_f, log_sigma, 1)
    c_sigma = np.exp(log_c_sigma)
    
    return P2040Parameters(
        name=material.name,
        a=a_eps,
        b=b_eps,
        c=c_sigma,
        d=d_sigma,
        notes=f"σ_f = {sigma_f:.0f} rayls/m, Delany-Bazley model"
    )


def empirical_material_to_p2040(material: AcousticMaterial) -> P2040Parameters:
    """
    Convert empirical acoustic material to P.2040 parameters.
    
    For complex objects (furniture, people) where physical modeling is
    impractical, we use measured absorption coefficients (Sabine α).
    
    The absorption coefficient α relates to reflection coefficient R by:
        α = 1 - |R|²
        |R| = √(1 - α)
    
    For a dielectric interface:
        R = (1 - √ε) / (1 + √ε)
        
    We solve for ε that gives the correct |R|. For a lossy material:
        ε = ε' - j×ε''
        
    Strategy:
    1. From α, compute target |R|
    2. Find ε that produces this |R| while accounting for absorption
    3. Fit ε'(f) and σ(f) to power-law form
    """
    if material.alpha_coeffs is None or material.d_eff is None:
        raise ValueError(f"Empirical material {material.name} requires alpha_coeffs and d_eff")
    
    # Standard octave band frequencies (Hz)
    std_freqs_hz = [125, 250, 500, 1000, 2000, 4000]
    
    if len(material.alpha_coeffs) != len(std_freqs_hz):
        raise ValueError(f"alpha_coeffs must have {len(std_freqs_hz)} values")
    
    alpha_coeffs = material.alpha_coeffs
    d_eff = material.d_eff
    
    # For empirical materials, we need to find ε such that:
    # 1. The reflection coefficient magnitude matches |R| = √(1-α)
    # 2. The material appears absorptive (has some ε'')
    #
    # For an interface reflection: |R| = |1 - √ε| / |1 + √ε|
    #
    # If we assume √ε is real and < 1 (material "rarer" than air, like porous):
    #   R = (1 - √ε') / (1 + √ε')  [real, positive]
    #   √ε' = (1 - R) / (1 + R)
    #   ε' = [(1 - R) / (1 + R)]²
    #
    # The absorption not accounted for by reflection goes into ε''
    
    epsilon_r_values = []
    epsilon_i_values = []
    f_ghz_values = []
    
    for f_hz, alpha in zip(std_freqs_hz, alpha_coeffs):
        f_ghz = acoustic_freq_to_radio_freq(f_hz)
        f_ghz_values.append(f_ghz)
        
        # Target reflection coefficient magnitude
        R_target = np.sqrt(max(1 - alpha, 0.001))  # Avoid zero
        
        # For porous-like materials with ε' < 1:
        # R = (1 - √ε') / (1 + √ε')  →  √ε' = (1 - R) / (1 + R)
        sqrt_eps = (1 - R_target) / (1 + R_target)
        epsilon_r = sqrt_eps ** 2
        
        # Add some loss (ε'') to account for internal absorption
        # The fraction of energy absorbed internally is α - (1 - R²)
        # But for a simple model, we'll make ε'' proportional to α
        # This gives materials with higher absorption more loss
        epsilon_i = alpha * 0.5  # Empirical scaling factor
        
        # Ensure reasonable bounds
        epsilon_r = max(epsilon_r, 0.01)
        epsilon_r = min(epsilon_r, 1.0)
        epsilon_i = max(epsilon_i, 0.001)
        
        epsilon_r_values.append(epsilon_r)
        epsilon_i_values.append(epsilon_i)
    
    # Fit ε'(f) - for empirical materials, often nearly constant
    log_f = np.log(f_ghz_values)
    log_eps = np.log(epsilon_r_values)
    
    # Check if there's significant frequency dependence
    b_eps, log_a_eps = np.polyfit(log_f, log_eps, 1)
    a_eps = np.exp(log_a_eps)
    
    # If b is very small, use constant ε' (b=0)
    if abs(b_eps) < 0.1:
        a_eps = np.mean(epsilon_r_values)
        b_eps = 0.0
    
    # Convert ε'' to σ using P.2040 formula: σ = ε'' × f_ghz / 17.98
    sigma_values = [eps_i * f / 17.98 for eps_i, f in zip(epsilon_i_values, f_ghz_values)]
    
    # Fit σ(f) = c × f^d
    sigma_positive = [max(s, 1e-20) for s in sigma_values]
    log_sigma = np.log(sigma_positive)
    d_sigma, log_c_sigma = np.polyfit(log_f, log_sigma, 1)
    c_sigma = np.exp(log_c_sigma)
    
    return P2040Parameters(
        name=material.name,
        a=a_eps,
        b=b_eps,
        c=c_sigma,
        d=d_sigma,
        notes=f"Empirical fit from α coeffs, d_eff = {d_eff:.2f} m"
    )


def convert_material(material: AcousticMaterial) -> P2040Parameters:
    """
    Convert any acoustic material to P.2040 parameters.
    
    Dispatches to the appropriate conversion function based on material type.
    """
    if material.material_type == MaterialType.RIGID:
        return rigid_material_to_p2040(material)
    elif material.material_type == MaterialType.POROUS:
        return porous_material_to_p2040(material)
    elif material.material_type == MaterialType.EMPIRICAL:
        return empirical_material_to_p2040(material)
    else:
        raise ValueError(f"Unknown material type: {material.material_type}")


# =============================================================================
# Material Database
# =============================================================================

def create_material_database() -> List[AcousticMaterial]:
    """
    Create database of common acoustic materials.
    
    Returns a list of AcousticMaterial objects with typical properties.
    
    Sources:
    - Acoustic impedance: Various acoustic engineering references
    - Flow resistivity: Manufacturer data, Bies & Hansen
    - Absorption coefficients: Collected from acoustic measurement databases
    """
    materials = []
    
    # =========================================================================
    # Reference Medium: Air
    # =========================================================================
    
    materials.append(AcousticMaterial(
        name="air",
        material_type=MaterialType.RIGID,  # Treated as rigid for parameterization
        rho=RHO_AIR,   # 1.2 kg/m³
        c=C_SOUND      # 343 m/s
    ))

    # =========================================================================
    # Class 1: Rigid Reflectors
    # =========================================================================
    
    materials.append(AcousticMaterial(
        name="concrete",
        material_type=MaterialType.RIGID,
        rho=2400,  # kg/m³
        c=3500     # m/s
    ))
    
    materials.append(AcousticMaterial(
        name="glass",
        material_type=MaterialType.RIGID,
        rho=2500,
        c=5500
    ))
    
    materials.append(AcousticMaterial(
        name="gypsum_board",
        material_type=MaterialType.RIGID,
        rho=750,
        c=1800
    ))
    
    materials.append(AcousticMaterial(
        name="wood",
        material_type=MaterialType.RIGID,
        rho=700,
        c=4000
    ))
    
    materials.append(AcousticMaterial(
        name="plywood",
        material_type=MaterialType.RIGID,
        rho=600,
        c=2500
    ))
    
    materials.append(AcousticMaterial(
        name="steel",
        material_type=MaterialType.RIGID,
        rho=7800,
        c=5900
    ))
    
    materials.append(AcousticMaterial(
        name="brick",
        material_type=MaterialType.RIGID,
        rho=1800,
        c=3600
    ))
    
    materials.append(AcousticMaterial(
        name="marble",
        material_type=MaterialType.RIGID,
        rho=2700,
        c=3800
    ))
    
    # =========================================================================
    # Class 2: Porous Absorbers
    # =========================================================================
    
    materials.append(AcousticMaterial(
        name="foam_light_melamine",
        material_type=MaterialType.POROUS,
        flow_resistivity=10000,  # rayls/m
        thickness=0.050         # 50mm typical
    ))
    
    materials.append(AcousticMaterial(
        name="foam_medium_acoustic",
        material_type=MaterialType.POROUS,
        flow_resistivity=20000,
        thickness=0.075
    ))
    
    materials.append(AcousticMaterial(
        name="fiberglass_dense",
        material_type=MaterialType.POROUS,
        flow_resistivity=40000,
        thickness=0.050
    ))
    
    materials.append(AcousticMaterial(
        name="mineral_wool",
        material_type=MaterialType.POROUS,
        flow_resistivity=30000,
        thickness=0.050
    ))
    
    materials.append(AcousticMaterial(
        name="curtain_heavy_velour",
        material_type=MaterialType.POROUS,
        flow_resistivity=8000,
        thickness=0.020
    ))
    
    materials.append(AcousticMaterial(
        name="curtain_light_cotton",
        material_type=MaterialType.POROUS,
        flow_resistivity=3000,
        thickness=0.008
    ))
    
    materials.append(AcousticMaterial(
        name="carpet_thick",
        material_type=MaterialType.POROUS,
        flow_resistivity=20000,
        thickness=0.015
    ))
    
    materials.append(AcousticMaterial(
        name="carpet_thin",
        material_type=MaterialType.POROUS,
        flow_resistivity=10000,
        thickness=0.008
    ))
    
    # =========================================================================
    # Class 3: Empirical Absorbers (Furniture, People)
    # Absorption coefficients at [125, 250, 500, 1000, 2000, 4000] Hz
    # =========================================================================
    
    materials.append(AcousticMaterial(
        name="couch_fabric",
        material_type=MaterialType.EMPIRICAL,
        alpha_coeffs=[0.20, 0.35, 0.55, 0.70, 0.75, 0.70],
        d_eff=0.25
    ))
    
    materials.append(AcousticMaterial(
        name="couch_leather",
        material_type=MaterialType.EMPIRICAL,
        alpha_coeffs=[0.10, 0.15, 0.25, 0.35, 0.40, 0.40],
        d_eff=0.25
    ))
    
    materials.append(AcousticMaterial(
        name="mattress",
        material_type=MaterialType.EMPIRICAL,
        alpha_coeffs=[0.15, 0.30, 0.50, 0.65, 0.70, 0.70],
        d_eff=0.20
    ))
    
    materials.append(AcousticMaterial(
        name="chair_upholstered",
        material_type=MaterialType.EMPIRICAL,
        alpha_coeffs=[0.15, 0.25, 0.40, 0.55, 0.60, 0.55],
        d_eff=0.20
    ))
    
    materials.append(AcousticMaterial(
        name="chair_wood",
        material_type=MaterialType.EMPIRICAL,
        alpha_coeffs=[0.02, 0.03, 0.03, 0.04, 0.04, 0.04],
        d_eff=0.10
    ))
    
    materials.append(AcousticMaterial(
        name="table_wood",
        material_type=MaterialType.EMPIRICAL,
        alpha_coeffs=[0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        d_eff=0.05
    ))
    
    materials.append(AcousticMaterial(
        name="bookshelf_filled",
        material_type=MaterialType.EMPIRICAL,
        alpha_coeffs=[0.15, 0.25, 0.30, 0.35, 0.40, 0.45],
        d_eff=0.30
    ))
    
    materials.append(AcousticMaterial(
        name="person_standing",
        material_type=MaterialType.EMPIRICAL,
        alpha_coeffs=[0.25, 0.35, 0.45, 0.50, 0.50, 0.45],
        d_eff=0.30
    ))
    
    materials.append(AcousticMaterial(
        name="person_seated",
        material_type=MaterialType.EMPIRICAL,
        alpha_coeffs=[0.20, 0.30, 0.40, 0.45, 0.45, 0.40],
        d_eff=0.25
    ))
    
    materials.append(AcousticMaterial(
        name="audience_dense",
        material_type=MaterialType.EMPIRICAL,
        alpha_coeffs=[0.40, 0.55, 0.70, 0.80, 0.85, 0.80],
        d_eff=0.30
    ))
    
    materials.append(AcousticMaterial(
        name="audience_sparse",
        material_type=MaterialType.EMPIRICAL,
        alpha_coeffs=[0.25, 0.35, 0.50, 0.60, 0.65, 0.60],
        d_eff=0.30
    ))
    
    return materials


# =============================================================================
# Utility Functions
# =============================================================================

def print_frequency_mapping():
    """Print the acoustic to radio frequency mapping table."""
    print("\n" + "="*60)
    print("Frequency Mapping: Acoustic (Hz) → Radio (GHz)")
    print("="*60)
    print(f"{'Acoustic (Hz)':>15} {'Radio (GHz)':>15} {'λ (m)':>12}")
    print("-"*60)
    
    acoustic_freqs = [30, 63, 125, 250, 500, 1000, 2000, 4000, 8000]
    for f_hz in acoustic_freqs:
        f_ghz = acoustic_freq_to_radio_freq(f_hz)
        wavelength = C_SOUND / f_hz
        print(f"{f_hz:>15} {f_ghz:>15.4f} {wavelength:>12.3f}")


def print_p2040_table(params_list: List[P2040Parameters]):
    """Print P.2040 parameters in a formatted table."""
    print("\n" + "="*90)
    print("ITU-R P.2040 Acoustic Material Parameters")
    print("Model: ε'(f) = a × f^b,  σ(f) = c × f^d  (f in GHz)")
    print("="*90)
    print(f"{'Material':<25} {'a':>12} {'b':>8} {'c':>12} {'d':>8}")
    print("-"*90)
    
    for p in params_list:
        print(f"{p.name:<25} {p.a:>12.4e} {p.b:>8.4f} {p.c:>12.4e} {p.d:>8.4f}")


def export_to_json(params_list: List[P2040Parameters], filename: str):
    """Export P.2040 parameters to JSON file."""
    data = {
        "description": "ITU-R P.2040 parameters for acoustic materials",
        "model": "epsilon_r = a * f^b, sigma = c * f^d (f in GHz)",
        "frequency_range": {
            "min_ghz": 0.0262,
            "max_ghz": 7.0,
            "acoustic_min_hz": 30,
            "acoustic_max_hz": 8000
        },
        "materials": [p.to_dict() for p in params_list]
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nExported to {filename}")


def export_to_csv(params_list: List[P2040Parameters], filename: str):
    """Export P.2040 parameters to CSV file."""
    with open(filename, 'w') as f:
        f.write("name,a,b,c,d,att\n")
        for p in params_list:
            f.write(f"{p.name},{p.a:.6g},{p.b:.6g},{p.c:.6g},{p.d:.6g},0.0\n")
    
    print(f"Exported to {filename}")


def validate_material(material: AcousticMaterial, params: P2040Parameters):
    """
    Validate converted parameters by checking reflection coefficients.
    
    Computes expected acoustic reflection vs. P.2040 model prediction
    at several frequencies.
    
    Returns the RMS error in |R| across frequencies.
    """
    print(f"\nValidation: {material.name}")
    print("-" * 60)
    
    test_freqs_hz = [125, 250, 500, 1000, 2000, 4000]
    errors = []
    
    for f_hz in test_freqs_hz:
        f_ghz = acoustic_freq_to_radio_freq(f_hz)
        
        # P.2040 model values
        eps_r = params.epsilon_real(f_ghz)
        eps_i = params.epsilon_imag(f_ghz)
        eps_complex = complex(eps_r, -eps_i)
        
        # Fresnel reflection at normal incidence (air → material)
        # R = (1 - √ε) / (1 + √ε)
        sqrt_eps = np.sqrt(eps_complex)
        R_p2040 = (1 - sqrt_eps) / (1 + sqrt_eps)
        
        # Expected acoustic reflection
        if material.material_type == MaterialType.RIGID:
            R_acoustic = (material.Z - Z_AIR) / (material.Z + Z_AIR)
            R_expected = abs(R_acoustic)
        elif material.material_type == MaterialType.POROUS:
            Z_c, _ = delany_bazley(f_hz, material.flow_resistivity)
            R_acoustic = (Z_c - Z_AIR) / (Z_c + Z_AIR)
            R_expected = abs(R_acoustic)
        else:
            # For empirical, use α to estimate |R|
            freq_idx = {125: 0, 250: 1, 500: 2, 1000: 3, 2000: 4, 4000: 5}
            idx = freq_idx.get(f_hz, 3)
            alpha = material.alpha_coeffs[idx]
            R_expected = np.sqrt(1 - alpha)
        
        error = abs(R_expected - abs(R_p2040))
        errors.append(error)
        
        print(f"  {f_hz:>5} Hz: |R|_acoustic = {R_expected:.4f}, "
              f"|R|_P2040 = {abs(R_p2040):.4f}, "
              f"error = {error:.4f}")
    
    rms_error = np.sqrt(np.mean(np.array(errors)**2))
    print(f"  RMS Error in |R|: {rms_error:.4f}")
    return rms_error


# =============================================================================
# Main
# =============================================================================

def main():
    """Main function: convert all materials and export."""
    
    print("="*70)
    print("Acoustic Material to ITU-R P.2040 Parameter Converter")
    print("="*70)
    
    # Print frequency mapping
    print_frequency_mapping()
    
    # Create material database
    materials = create_material_database()
    print(f"\nLoaded {len(materials)} materials")
    
    # Convert all materials
    p2040_params = []
    for mat in materials:
        try:
            params = convert_material(mat)
            p2040_params.append(params)
        except Exception as e:
            print(f"Error converting {mat.name}: {e}")
    
    # Print results
    print_p2040_table(p2040_params)
    
    # Validate all materials
    print("\n" + "="*70)
    print("Validation Summary (comparing acoustic vs P.2040 reflection)")
    print("="*70)
    
    errors_by_type = {MaterialType.RIGID: [], MaterialType.POROUS: [], MaterialType.EMPIRICAL: []}
    
    for mat, params in zip(materials, p2040_params):
        rms_error = validate_material(mat, params)
        errors_by_type[mat.material_type].append((mat.name, rms_error))
    
    # Print summary by material type
    print("\n" + "="*70)
    print("Error Summary by Material Type")
    print("="*70)
    for mat_type, errors in errors_by_type.items():
        if errors:
            avg_error = np.mean([e[1] for e in errors])
            max_error = max([e[1] for e in errors])
            worst_mat = max(errors, key=lambda x: x[1])[0]
            print(f"\n{mat_type.value.upper()} materials:")
            print(f"  Average RMS error: {avg_error:.4f}")
            print(f"  Maximum RMS error: {max_error:.4f} ({worst_mat})")
    
    # Export
    export_to_json(p2040_params, "materials_ac.json")
    export_to_csv(p2040_params, "materials_ac.csv")
    
    print("\n" + "="*70)
    print("Done!")
    print("="*70)
    
    return materials, p2040_params


if __name__ == "__main__":
    main()
