"""
Nexum v2.0 Robotic Body Design
Physical structure and mechanical design framework
"""

import numpy as np
import math
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

from ..kinematics.motor_controller import MotorConfig, MotorType


class MaterialType(Enum):
    CARBON_FIBER = "carbon_fiber"
    ALUMINUM = "aluminum"
    TITANIUM = "titanium"
    PLASTIC_ABS = "plastic_abs"
    PLASTIC_PETG = "plastic_petg"
    STEEL = "steel"
    COMPOSITE = "composite"


class BodySection(Enum):
    HEAD = "head"
    TORSO = "torso"
    LEFT_ARM = "left_arm"
    RIGHT_ARM = "right_arm"
    LEFT_LEG = "left_leg"
    RIGHT_LEG = "right_leg"
    WAIST = "waist"
    NECK = "neck"


@dataclass
class MaterialProperties:
    name: str
    density: float  # kg/m³
    youngs_modulus: float  # GPa
    yield_strength: float  # MPa
    thermal_conductivity: float  # W/(m·K)
    cost_factor: float  # Relative cost multiplier
    printability: bool  # Can be 3D printed
    weight_factor: float  # Weight efficiency (lower is better)


@dataclass
class ComponentPlacement:
    component_id: str
    component_type: str
    position: Tuple[float, float, float]  # x, y, z in meters
    rotation: Tuple[float, float, float]  # roll, pitch, yaw in degrees
    mounting_points: List[Tuple[float, float, float]]
    weight: float  # kg
    dimensions: Tuple[float, float, float]  # width, height, depth in meters
    cooling_requirement: float  # Watts
    vibration_sensitivity: float  # 0-1 scale


@dataclass
class BodySegment:
    section: BodySection
    material: MaterialType
    dimensions: Tuple[float, float, float]  # width, height, depth in meters
    wall_thickness: float  # meters
    internal_components: List[ComponentPlacement]
    mounting_holes: List[Tuple[float, float, float, float]]  # x, y, z, diameter
    weight: float  # kg
    center_of_mass: Tuple[float, float, float]


class MaterialDatabase:
    """Database of material properties for robotic construction"""
    
    def __init__(self):
        self.materials = {
            MaterialType.CARBON_FIBER: MaterialProperties(
                name="Carbon Fiber",
                density=1600,
                youngs_modulus=230,
                yield_strength=3500,
                thermal_conductivity=8.0,
                cost_factor=3.0,
                printability=False,
                weight_factor=0.4
            ),
            MaterialType.ALUMINUM: MaterialProperties(
                name="Aluminum 6061",
                density=2700,
                youngs_modulus=68.9,
                yield_strength=276,
                thermal_conductivity=167,
                cost_factor=1.0,
                printability=False,
                weight_factor=0.7
            ),
            MaterialType.TITANIUM: MaterialProperties(
                name="Titanium Ti-6Al-4V",
                density=4430,
                youngs_modulus=113.8,
                yield_strength=880,
                thermal_conductivity=6.7,
                cost_factor=5.0,
                printability=False,
                weight_factor=0.9
            ),
            MaterialType.PLASTIC_ABS: MaterialProperties(
                name="ABS Plastic",
                density=1050,
                youngs_modulus=2.3,
                yield_strength=40,
                thermal_conductivity=0.25,
                cost_factor=0.3,
                printability=True,
                weight_factor=0.3
            ),
            MaterialType.PLASTIC_PETG: MaterialProperties(
                name="PETG Plastic",
                density=1270,
                youngs_modulus=2.1,
                yield_strength=50,
                thermal_conductivity=0.29,
                cost_factor=0.4,
                printability=True,
                weight_factor=0.4
            ),
            MaterialType.STEEL: MaterialProperties(
                name="Stainless Steel 304",
                density=8000,
                youngs_modulus=193,
                yield_strength=520,
                thermal_conductivity=16.2,
                cost_factor=1.5,
                printability=False,
                weight_factor=1.0
            ),
            MaterialType.COMPOSITE: MaterialProperties(
                name="Composite Material",
                density=1800,
                youngs_modulus=150,
                yield_strength=800,
                thermal_conductivity=5.0,
                cost_factor=2.5,
                printability=False,
                weight_factor=0.5
            )
        }
    
    def get_material(self, material_type: MaterialType) -> MaterialProperties:
        """Get material properties"""
        return self.materials.get(material_type, self.materials[MaterialType.ALUMINUM])
    
    def recommend_material(self, requirements: Dict[str, Any]) -> MaterialType:
        """Recommend material based on requirements"""
        strength_required = requirements.get('strength', 0)
        weight_limit = requirements.get('weight_limit', float('inf'))
        thermal_requirement = requirements.get('thermal', 0)
        printable = requirements.get('printable', False)
        budget_factor = requirements.get('budget', 1.0)
        
        best_material = MaterialType.ALUMINUM
        best_score = 0
        
        for material_type, properties in self.materials.items():
            score = 0
            
            # Strength score
            if properties.yield_strength >= strength_required:
                score += 2
            else:
                score -= 1
            
            # Weight score
            if properties.density * 0.001 <= weight_limit:  # Convert to g/cm³
                score += 2
            else:
                score -= 1
            
            # Thermal score
            if properties.thermal_conductivity >= thermal_requirement:
                score += 1
            
            # Printability score
            if printable and properties.printability:
                score += 2
            elif not printable:
                score += 1
            
            # Cost score
            if properties.cost_factor <= budget_factor:
                score += 1
            
            if score > best_score:
                best_score = score
                best_material = material_type
        
        return best_material


class BodyDesigner:
    """
    Robotic body design system for Nexum v2.0
    Handles structural design, material selection, and component placement
    """
    
    def __init__(self):
        self.material_db = MaterialDatabase()
        self.logger = logging.getLogger(__name__)
        
        # Body segments
        self.segments: Dict[BodySection, BodySegment] = {}
        
        # Design parameters
        self.total_height = 1.75  # meters (human-like)
        self.total_weight = 0  # Will be calculated
        self.center_of_mass = (0, 0, 0)
        
        # Component library
        self.component_library = self._create_component_library()
    
    def _create_component_library(self) -> Dict[str, ComponentPlacement]:
        """Create library of standard components"""
        return {
            "main_cpu": ComponentPlacement(
                component_id="main_cpu",
                component_type="processor",
                position=(0, 0.1, 0),
                rotation=(0, 0, 0),
                mounting_points=[(-0.05, 0, 0), (0.05, 0, 0)],
                weight=2.0,
                dimensions=(0.1, 0.05, 0.15),
                cooling_requirement=65,
                vibration_sensitivity=0.3
            ),
            "gpu_unit": ComponentPlacement(
                component_id="gpu_unit",
                component_type="graphics",
                position=(0, 0.05, 0.1),
                rotation=(0, 0, 0),
                mounting_points=[(-0.08, 0, 0.1), (0.08, 0, 0.1)],
                weight=1.5,
                dimensions=(0.16, 0.1, 0.25),
                cooling_requirement=250,
                vibration_sensitivity=0.4
            ),
            "battery_pack": ComponentPlacement(
                component_id="battery_pack",
                component_type="power",
                position=(0, -0.1, 0),
                rotation=(0, 0, 0),
                mounting_points=[(-0.06, -0.1, 0), (0.06, -0.1, 0)],
                weight=3.0,
                dimensions=(0.12, 0.08, 0.2),
                cooling_requirement=10,
                vibration_sensitivity=0.1
            ),
            "audio_processor": ComponentPlacement(
                component_id="audio_processor",
                component_type="audio",
                position=(0, 0.15, 0.2),
                rotation=(0, 0, 0),
                mounting_points=[(-0.03, 0.15, 0.2), (0.03, 0.15, 0.2)],
                weight=0.5,
                dimensions=(0.06, 0.04, 0.08),
                cooling_requirement=5,
                vibration_sensitivity=0.2
            ),
            "motor_controller": ComponentPlacement(
                component_id="motor_controller",
                component_type="control",
                position=(0, 0, -0.1),
                rotation=(0, 0, 0),
                mounting_points=[(-0.04, 0, -0.1), (0.04, 0, -0.1)],
                weight=0.8,
                dimensions=(0.08, 0.05, 0.12),
                cooling_requirement=15,
                vibration_sensitivity=0.1
            )
        }
    
    def design_humanoid_body(self, height: float = 1.75, weight_target: float = 80) -> Dict[str, Any]:
        """Design a humanoid robotic body"""
        self.total_height = height
        
        # Calculate segment proportions based on human anatomy
        head_height = height * 0.125
        neck_height = height * 0.05
        torso_height = height * 0.3
        waist_height = height * 0.08
        leg_height = height * 0.4
        arm_length = height * 0.35
        
        # Design each segment
        self._design_head(head_height)
        self._design_neck(neck_height)
        self._design_torso(torso_height)
        self._design_waist(waist_height)
        self._design_legs(leg_height)
        self._design_arms(arm_length)
        
        # Calculate total weight
        self._calculate_weight_distribution()
        
        # Optimize for weight target
        if self.total_weight > weight_target:
            self._optimize_weight(weight_target)
        
        return self.get_design_summary()
    
    def _design_head(self, height: float):
        """Design head segment"""
        width = height * 0.8
        depth = height * 0.9
        
        # Material selection
        requirements = {
            'strength': 50,
            'weight_limit': 2.0,
            'thermal': 1.0,
            'printable': True
        }
        material = self.material_db.recommend_material(requirements)
        
        # Components in head
        components = [
            self.component_library["audio_processor"],
            ComponentPlacement(
                component_id="cameras",
                component_type="vision",
                position=(0, height/2, depth/2),
                rotation=(0, -15, 0),
                mounting_points=[(-0.03, height/2, depth/2), (0.03, height/2, depth/2)],
                weight=0.3,
                dimensions=(0.06, 0.04, 0.05),
                cooling_requirement=8,
                vibration_sensitivity=0.6
            )
        ]
        
        self.segments[BodySection.HEAD] = BodySegment(
            section=BodySection.HEAD,
            material=material,
            dimensions=(width, height, depth),
            wall_thickness=0.003,
            internal_components=components,
            mounting_holes=[
                (width/2, 0, 0, 0.005),  # Neck attachment
                (-width/2, 0, 0, 0.005)
            ],
            weight=0,
            center_of_mass=(0, 0, 0)
        )
    
    def _design_neck(self, height: float):
        """Design neck segment"""
        width = height * 0.6
        depth = height * 0.7
        
        requirements = {
            'strength': 200,
            'weight_limit': 1.5,
            'thermal': 5.0,
            'printable': False
        }
        material = self.material_db.recommend_material(requirements)
        
        self.segments[BodySection.NECK] = BodySegment(
            section=BodySection.NECK,
            material=material,
            dimensions=(width, height, depth),
            wall_thickness=0.005,
            internal_components=[],
            mounting_holes=[
                (width/2, height/2, 0, 0.006),  # Head attachment
                (-width/2, height/2, 0, 0.006),
                (width/2, -height/2, 0, 0.006),  # Torso attachment
                (-width/2, -height/2, 0, 0.006)
            ],
            weight=0,
            center_of_mass=(0, 0, 0)
        )
    
    def _design_torso(self, height: float):
        """Design torso segment"""
        width = height * 0.4
        depth = height * 0.25
        
        requirements = {
            'strength': 300,
            'weight_limit': 15.0,
            'thermal': 10.0,
            'printable': False
        }
        material = self.material_db.recommend_material(requirements)
        
        # Main components in torso
        components = [
            self.component_library["main_cpu"],
            self.component_library["gpu_unit"],
            self.component_library["battery_pack"],
            self.component_library["motor_controller"]
        ]
        
        # Add mounting points for arms
        mounting_holes = [
            (width/2, height/4, 0, 0.008),   # Right arm
            (-width/2, height/4, 0, 0.008),  # Left arm
            (width/2, -height/4, 0, 0.008),  # Right leg
            (-width/2, -height/4, 0, 0.008), # Left leg
            (width/2, height/2, 0, 0.006),   # Neck attachment
            (-width/2, height/2, 0, 0.006),
            (width/2, -height/2, 0, 0.006),  # Waist attachment
            (-width/2, -height/2, 0, 0.006)
        ]
        
        self.segments[BodySection.TORSO] = BodySegment(
            section=BodySection.TORSO,
            material=material,
            dimensions=(width, height, depth),
            wall_thickness=0.006,
            internal_components=components,
            mounting_holes=mounting_holes,
            weight=0,
            center_of_mass=(0, 0, 0)
        )
    
    def _design_waist(self, height: float):
        """Design waist segment"""
        width = height * 0.5
        depth = height * 0.3
        
        requirements = {
            'strength': 250,
            'weight_limit': 3.0,
            'thermal': 5.0,
            'printable': False
        }
        material = self.material_db.recommend_material(requirements)
        
        self.segments[BodySection.WAIST] = BodySegment(
            section=BodySection.WAIST,
            material=material,
            dimensions=(width, height, depth),
            wall_thickness=0.004,
            internal_components=[],
            mounting_holes=[
                (width/2, height/2, 0, 0.006),  # Torso attachment
                (-width/2, height/2, 0, 0.006),
                (width/2, -height/2, 0, 0.006),  # Legs attachment
                (-width/2, -height/2, 0, 0.006)
            ],
            weight=0,
            center_of_mass=(0, 0, 0)
        )
    
    def _design_legs(self, height: float):
        """Design leg segments"""
        # Upper leg (thigh)
        upper_width = height * 0.15
        upper_depth = height * 0.12
        
        requirements = {
            'strength': 400,
            'weight_limit': 8.0,
            'thermal': 3.0,
            'printable': False
        }
        material = self.material_db.recommend_material(requirements)
        
        # Lower leg (calf)
        lower_width = height * 0.1
        lower_depth = height * 0.08
        
        self.segments[BodySection.LEFT_LEG] = BodySegment(
            section=BodySection.LEFT_LEG,
            material=material,
            dimensions=(upper_width, height, upper_depth),
            wall_thickness=0.005,
            internal_components=[],
            mounting_holes=[
                (upper_width/2, height/2, 0, 0.008),  # Waist attachment
                (-upper_width/2, height/2, 0, 0.008),
                (lower_width/2, -height/2, 0, 0.006),  # Foot attachment
                (-lower_width/2, -height/2, 0, 0.006)
            ],
            weight=0,
            center_of_mass=(0, 0, 0)
        )
        
        self.segments[BodySection.RIGHT_LEG] = BodySegment(
            section=BodySection.RIGHT_LEG,
            material=material,
            dimensions=(upper_width, height, upper_depth),
            wall_thickness=0.005,
            internal_components=[],
            mounting_holes=[
                (upper_width/2, height/2, 0, 0.008),  # Waist attachment
                (-upper_width/2, height/2, 0, 0.008),
                (lower_width/2, -height/2, 0, 0.006),  # Foot attachment
                (-lower_width/2, -height/2, 0, 0.006)
            ],
            weight=0,
            center_of_mass=(0, 0, 0)
        )
    
    def _design_arms(self, length: float):
        """Design arm segments"""
        # Upper arm
        upper_width = length * 0.12
        upper_depth = length * 0.1
        
        requirements = {
            'strength': 200,
            'weight_limit': 4.0,
            'thermal': 2.0,
            'printable': True
        }
        material = self.material_db.recommend_material(requirements)
        
        # Lower arm (forearm)
        lower_width = length * 0.08
        lower_depth = length * 0.06
        
        self.segments[BodySection.LEFT_ARM] = BodySegment(
            section=BodySection.LEFT_ARM,
            material=material,
            dimensions=(upper_width, length, upper_depth),
            wall_thickness=0.004,
            internal_components=[],
            mounting_holes=[
                (upper_width/2, length/2, 0, 0.006),  # Shoulder attachment
                (-upper_width/2, length/2, 0, 0.006),
                (lower_width/2, -length/2, 0, 0.005),  # Hand attachment
                (-lower_width/2, -length/2, 0, 0.005)
            ],
            weight=0,
            center_of_mass=(0, 0, 0)
        )
        
        self.segments[BodySection.RIGHT_ARM] = BodySegment(
            section=BodySection.RIGHT_ARM,
            material=material,
            dimensions=(upper_width, length, upper_depth),
            wall_thickness=0.004,
            internal_components=[],
            mounting_holes=[
                (upper_width/2, length/2, 0, 0.006),  # Shoulder attachment
                (-upper_width/2, length/2, 0, 0.006),
                (lower_width/2, -length/2, 0, 0.005),  # Hand attachment
                (-lower_width/2, -length/2, 0, 0.005)
            ],
            weight=0,
            center_of_mass=(0, 0, 0)
        )
    
    def _calculate_weight_distribution(self):
        """Calculate weight and center of mass for all segments"""
        total_weight = 0
        total_moment = np.array([0.0, 0.0, 0.0])
        
        for section, segment in self.segments.items():
            # Calculate segment weight
            material = self.material_db.get_material(segment.material)
            volume = (segment.dimensions[0] * segment.dimensions[1] * segment.dimensions[2] -
                     (segment.dimensions[0] - 2*segment.wall_thickness) * 
                     (segment.dimensions[1] - 2*segment.wall_thickness) * 
                     (segment.dimensions[2] - 2*segment.wall_thickness))
            
            segment_weight = volume * material.density
            segment.weight = segment_weight
            
            # Add component weights
            for component in segment.internal_components:
                segment_weight += component.weight
            
            total_weight += segment_weight
            
            # Calculate center of mass contribution
            segment_com = np.array(segment.center_of_mass)
            total_moment += segment_com * segment_weight
            
            # Update segment weight
            self.segments[section] = segment
        
        self.total_weight = total_weight
        self.center_of_mass = tuple(total_moment / total_weight)
    
    def _optimize_weight(self, target_weight: float):
        """Optimize design to meet weight target"""
        current_weight = self.total_weight
        
        if current_weight <= target_weight:
            return
        
        # Calculate weight reduction needed
        weight_reduction = current_weight - target_weight
        
        # Strategy 1: Reduce wall thickness
        thickness_reduction = weight_reduction / (current_weight * 0.3)
        
        for section, segment in self.segments.items():
            new_thickness = max(0.002, segment.wall_thickness - thickness_reduction)
            segment.wall_thickness = new_thickness
            self.segments[section] = segment
        
        # Recalculate weight
        self._calculate_weight_distribution()
        
        # Strategy 2: Change to lighter materials if still overweight
        if self.total_weight > target_weight:
            for section, segment in self.segments.items():
                current_material = self.material_db.get_material(segment.material)
                
                # Find lighter alternative
                lighter_material = None
                for material_type, properties in self.material_db.materials.items():
                    if (properties.density < current_material.density and
                        properties.yield_strength >= current_material.yield_strength * 0.8):
                        lighter_material = material_type
                        break
                
                if lighter_material:
                    segment.material = lighter_material
                    self.segments[section] = segment
        
        # Final recalculation
        self._calculate_weight_distribution()
    
    def get_design_summary(self) -> Dict[str, Any]:
        """Get complete design summary"""
        return {
            "total_height": self.total_height,
            "total_weight": self.total_weight,
            "center_of_mass": self.center_of_mass,
            "segments": {
                section.value: {
                    "material": segment.material.value,
                    "dimensions": segment.dimensions,
                    "wall_thickness": segment.wall_thickness,
                    "weight": segment.weight,
                    "components_count": len(segment.internal_components),
                    "material_properties": self.material_db.get_material(segment.material).__dict__
                }
                for section, segment in self.segments.items()
            },
            "material_usage": self._get_material_usage(),
            "component_summary": self._get_component_summary()
        }
    
    def _get_material_usage(self) -> Dict[str, float]:
        """Get material usage summary"""
        usage = {}
        for segment in self.segments.values():
            material = segment.material.value
            if material not in usage:
                usage[material] = 0
            usage[material] += segment.weight
        
        return usage
    
    def _get_component_summary(self) -> Dict[str, Any]:
        """Get component placement summary"""
        all_components = []
        total_component_weight = 0
        total_cooling_requirement = 0
        
        for segment in self.segments.values():
            for component in segment.internal_components:
                all_components.append({
                    "id": component.component_id,
                    "type": component.component_type,
                    "weight": component.weight,
                    "cooling": component.cooling_requirement,
                    "vibration_sensitivity": component.vibration_sensitivity
                })
                total_component_weight += component.weight
                total_cooling_requirement += component.cooling_requirement
        
        return {
            "total_components": len(all_components),
            "total_weight": total_component_weight,
            "total_cooling_requirement": total_cooling_requirement,
            "components": all_components
        }
    
    def export_design(self, filename: str):
        """Export design to JSON file"""
        design_data = self.get_design_summary()
        
        with open(filename, 'w') as f:
            json.dump(design_data, f, indent=2)
        
        self.logger.info(f"Design exported to {filename}")
    
    def generate_manufacturing_files(self, output_dir: str):
        """Generate manufacturing files (3D printing, CNC, etc.)"""
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate STL files for 3D printable parts
        for section, segment in self.segments.items():
            if self.material_db.get_material(segment.material).printability:
                stl_filename = f"{output_dir}/{section.value}_segment.stl"
                self._generate_stl_file(segment, stl_filename)
        
        # Generate CNC files for metal parts
        for section, segment in self.segments.items():
            if not self.material_db.get_material(segment.material).printability:
                cnc_filename = f"{output_dir}/{section.value}_segment.cnc"
                self._generate_cnc_file(segment, cnc_filename)
        
        # Generate assembly instructions
        assembly_filename = f"{output_dir}/assembly_instructions.md"
        self._generate_assembly_instructions(assembly_filename)
        
        self.logger.info(f"Manufacturing files generated in {output_dir}")
    
    def _generate_stl_file(self, segment: BodySegment, filename: str):
        """Generate STL file for 3D printing"""
        # This would integrate with a CAD library to generate actual STL files
        # For now, create a placeholder with dimensions
        stl_content = f"""# STL File for {segment.section.value}
# Dimensions: {segment.dimensions}
# Wall Thickness: {segment.wall_thickness}
# Material: {segment.material.value}
# Weight: {segment.weight:.2f} kg

# This is a placeholder STL file
# In production, this would contain actual mesh data
"""
        
        with open(filename, 'w') as f:
            f.write(stl_content)
    
    def _generate_cnc_file(self, segment: BodySegment, filename: str):
        """Generate CNC file for metal parts"""
        cnc_content = f"""# CNC File for {segment.section.value}
# Dimensions: {segment.dimensions}
# Wall Thickness: {segment.wall_thickness}
# Material: {segment.material.value}
# Weight: {segment.weight:.2f} kg

# This is a placeholder CNC file
# In production, this would contain actual G-code
"""
        
        with open(filename, 'w') as f:
            f.write(cnc_content)
    
    def _generate_assembly_instructions(self, filename: str):
        """Generate assembly instructions"""
        instructions = """# Nexum v2.0 Assembly Instructions

## Assembly Order
1. **Torso Core** - Install main components (CPU, GPU, Battery)
2. **Head Unit** - Attach to torso via neck segment
3. **Arms** - Attach to torso shoulders
4. **Legs** - Attach to torso via waist segment
5. **Final Integration** - Connect all systems

## Critical Tolerances
- Mounting hole tolerance: ±0.1mm
- Component alignment: ±0.5mm
- Weight distribution: Maintain center of mass within 10cm of design point

## Torque Specifications
- M4 screws: 2.5 Nm
- M6 screws: 8 Nm
- M8 screws: 20 Nm

## Safety Notes
- Verify all electrical connections before power-on
- Check cooling system functionality
- Test emergency stop systems
- Verify motor calibration
"""
        
        with open(filename, 'w') as f:
            f.write(instructions)


def main():
    """Example usage of body designer"""
    designer = BodyDesigner()
    
    # Design humanoid body
    design = designer.design_humanoid_body(height=1.75, weight_target=75)
    
    print("=== Nexum v2.0 Body Design ===")
    print(f"Total Height: {design['total_height']:.2f} m")
    print(f"Total Weight: {design['total_weight']:.2f} kg")
    print(f"Center of Mass: {design['center_of_mass']}")
    
    print("\n=== Material Usage ===")
    for material, weight in design['material_usage'].items():
        print(f"{material}: {weight:.2f} kg")
    
    print("\n=== Component Summary ===")
    comp_summary = design['component_summary']
    print(f"Total Components: {comp_summary['total_components']}")
    print(f"Component Weight: {comp_summary['total_weight']:.2f} kg")
    print(f"Cooling Requirement: {comp_summary['total_cooling_requirement']:.0f} W")
    
    # Export design
    designer.export_design("nexum_body_design.json")
    designer.generate_manufacturing_files("manufacturing_files")


if __name__ == "__main__":
    main()
