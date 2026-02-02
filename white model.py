import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min

def oasis_white_model(total_daily_kwh, u_value=0.5, temp_out=30, temp_set=22, 
                      w_shape=5.0, w_scale=14.0, ai_dampening=1.0):
    """
    total_daily_kwh: Input from 3-year data averages
    w_shape (k): Skewness of occupancy (High k = more symmetrical)
    w_scale (c): Peak time of occupancy
    ai_dampening: AI factor (0 to 1) to reduce occupancy influence for certain POIs
    """
    t = np.linspace(0, 24, 100)
    
    # 1. Weibull Distribution for Occupancy
    # Represents the 'Living Spot' dynamism better than a parabola
    occupancy_raw = weibull_min.pdf(t, w_shape, loc=0, scale=w_scale)
    occupancy = (occupancy_raw / np.max(occupancy_raw)) * ai_dampening
    
    # 2. Floor Area Calculation
    # Derived from total energy: Area = Total_kWh / (Estimated Avg intensity * 24h)
    # Intensity placeholder: 0.05 kWh/m2/h (typical for mid-range office/school)
    avg_intensity = 0.05 
    floor_area = total_daily_kwh / (avg_intensity * 24)
    
    # 3. Physics-Based Lighting
    # Lighting = Floor Area * Illumination (avg 10W/m2) * Control (0.8) * Occupancy
    avg_illumination_per_m2 = 0.010 # kW/m2
    control_factor = 0.8
    lighting_active = floor_area * avg_illumination_per_m2 * control_factor * occupancy
    lighting = (lighting_active * 0.85) + (lighting_active.max() * 0.15) # 15% Residual

    # 4. Ventilation Logic (Occupancy Driven 1 to 10 L/s)
    # Thermal component: (delT / COP)
    del_t = abs(temp_out - temp_set)
    cop_vent = 3.5 # Optimized for AI later
    vent_per_person = (occupancy * 9 + 1) # L/s per person scale
    vent_base = vent_per_person * (floor_area / 15) # Assuming 15m2 per person
    ventilation = (vent_base * 0.95) + (vent_base.max() * 0.05) + (del_t / cop_vent)

    # 5. HVAC (Building Physics)
    # Q = U * A * delT
    hvac_load_max = (u_value * floor_area * del_t) / 1000 # kW
    hvac = (occupancy * 0.95 * hvac_load_max) + (0.05 * hvac_load_max)

    # 6. Others (Plug Loads)
    others = (occupancy * 0.95 * (total_daily_kwh/24 * 0.2)) + (0.05 * (total_daily_kwh/24 * 0.2))

    return t, lighting, ventilation, hvac, others, floor_area

# Scenario: Large Office/School consuming 5000 kWh/day
t, light, vent, hvac, misc, area = oasis_white_model(total_daily_kwh=5000)

print(f"Calculated Floor Area: {area:.2f} m2")

# Visualization
plt.figure(figsize=(12, 6))
plt.plot(t, light, label='Lighting (Physics-Based)', color='gold', lw=2)
plt.plot(t, vent, label='Ventilation (L/s Scale)', color='skyblue', lw=2)
plt.plot(t, hvac, label='HVAC Load (U-Value Driven)', color='salmon', lw=2)
plt.plot(t, misc, label='Other Loads (Residual 5%)', color='gray', ls='--')

plt.title(f"Oasis White Model: Load Distribution for {area:.0f}m2 POI")
plt.xlabel("Time (24h)")
plt.ylabel("Power (kW)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
