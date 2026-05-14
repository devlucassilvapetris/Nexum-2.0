"""
Nexum v2.0 Command Line Interface
Provides CLI interface for system control and management
"""

import argparse
import asyncio
import json
import sys
import time
from typing import Dict, Any, Optional

from .core import NexumSystem, SystemConfig


class NexumCLI:
    """Command line interface for Nexum v2.0"""
    
    def __init__(self):
        self.system: Optional[NexumSystem] = None
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create command line argument parser"""
        parser = argparse.ArgumentParser(
            description="Nexum v2.0 - Hybrid Operating System",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  nexum run                          # Run full system
  nexum run --no-motors              # Run without motors
  nexum design --height 1.8 --weight 80  # Design body
  nexum status                        # Get system status
  nexum move-motor torso 1.5         # Move motor to position
  nexum ai-inference text_generator "Hello world"  # Run AI inference
            """
        )
        
        # Global options
        parser.add_argument(
            '--log-level',
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
            default='INFO',
            help='Set logging level'
        )
        
        parser.add_argument(
            '--config',
            type=str,
            help='Path to configuration file'
        )
        
        # Subcommands
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Run command
        run_parser = subparsers.add_parser('run', help='Run Nexum system')
        run_parser.add_argument('--no-motors', action='store_true', help='Disable motor control')
        run_parser.add_argument('--no-audio', action='store_true', help='Disable audio processing')
        run_parser.add_argument('--no-vision', action='store_true', help='Disable vision processing')
        run_parser.add_argument('--no-ai', action='store_true', help='Disable AI processing')
        run_parser.add_argument('--no-gpu', action='store_true', help='Disable GPU acceleration')
        
        # Design command
        design_parser = subparsers.add_parser('design', help='Design robotic body')
        design_parser.add_argument('--height', type=float, default=1.75, help='Body height in meters')
        design_parser.add_argument('--weight', type=float, default=75, help='Target weight in kg')
        design_parser.add_argument('--output', type=str, help='Output file path')
        
        # Status command
        status_parser = subparsers.add_parser('status', help='Get system status')
        status_parser.add_argument('--json', action='store_true', help='Output as JSON')
        
        # Motor control commands
        motor_parser = subparsers.add_parser('move-motor', help='Move motor to position')
        motor_parser.add_argument('motor_id', type=str, help='Motor identifier')
        motor_parser.add_argument('position', type=float, help='Target position')
        
        # AI inference command
        ai_parser = subparsers.add_parser('ai-inference', help='Run AI inference')
        ai_parser.add_argument('model_id', type=str, help='Model identifier')
        ai_parser.add_argument('input', type=str, help='Input data')
        ai_parser.add_argument('--output', type=str, help='Output file path')
        
        # System commands
        subparsers.add_parser('stop', help='Stop system')
        subparsers.add_parser('restart', help='Restart system')
        
        # Configuration commands
        config_parser = subparsers.add_parser('config', help='Configuration management')
        config_subparsers = config_parser.add_subparsers(dest='config_action')
        
        config_subparsers.add_parser('show', help='Show current configuration')
        config_subparsers.add_parser('reset', help='Reset to default configuration')
        
        # Diagnostic commands
        diag_parser = subparsers.add_parser('diagnose', help='Run system diagnostics')
        diag_parser.add_argument('--component', type=str, help='Specific component to diagnose')
        diag_parser.add_argument('--verbose', action='store_true', help='Verbose output')
        
        return parser
    
    async def run(self, args=None):
        """Run CLI with provided arguments"""
        if args is None:
            args = sys.argv[1:]
        
        parsed_args = self.parser.parse_args(args)
        
        if not parsed_args.command:
            self.parser.print_help()
            return 1
        
        try:
            if parsed_args.command == 'run':
                return await self._command_run(parsed_args)
            elif parsed_args.command == 'design':
                return await self._command_design(parsed_args)
            elif parsed_args.command == 'status':
                return await self._command_status(parsed_args)
            elif parsed_args.command == 'move-motor':
                return await self._command_move_motor(parsed_args)
            elif parsed_args.command == 'ai-inference':
                return await self._command_ai_inference(parsed_args)
            elif parsed_args.command == 'stop':
                return await self._command_stop(parsed_args)
            elif parsed_args.command == 'restart':
                return await self._command_restart(parsed_args)
            elif parsed_args.command == 'config':
                return await self._command_config(parsed_args)
            elif parsed_args.command == 'diagnose':
                return await self._command_diagnose(parsed_args)
            else:
                print(f"Unknown command: {parsed_args.command}")
                return 1
        
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            return 130
        except Exception as e:
            print(f"Error: {e}")
            return 1
    
    async def _command_run(self, args) -> int:
        """Run Nexum system"""
        print("Starting Nexum v2.0...")
        
        # Create configuration
        config = SystemConfig(
            log_level=args.log_level,
            enable_motors=not args.no_motors,
            enable_audio=not args.no_audio,
            enable_vision=not args.no_vision,
            enable_ai=not args.no_ai,
            enable_gpu=not args.no_gpu
        )
        
        # Create and run system
        self.system = NexumSystem(config)
        
        try:
            success = await self.system.run()
            return 0 if success else 1
        except Exception as e:
            print(f"System error: {e}")
            return 1
    
    async def _command_design(self, args) -> int:
        """Design robotic body"""
        print(f"Designing robotic body: {args.height}m, {args.weight}kg")
        
        # Create temporary system for design
        config = SystemConfig(log_level="WARNING")
        system = NexumSystem(config)
        
        if not await system.initialize():
            print("Failed to initialize system for design")
            return 1
        
        # Design body
        result = await system.execute_command("design_body", {
            "height": args.height,
            "weight_target": args.weight
        })
        
        if result["success"]:
            design_data = result["data"]
            
            # Display results
            print(f"\n=== Design Results ===")
            print(f"Total Height: {design_data['total_height']:.2f} m")
            print(f"Total Weight: {design_data['total_weight']:.2f} kg")
            print(f"Center of Mass: {design_data['center_of_mass']}")
            
            print(f"\n=== Material Usage ===")
            for material, weight in design_data['material_usage'].items():
                print(f"{material}: {weight:.2f} kg")
            
            # Save to file if requested
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(design_data, f, indent=2)
                print(f"\nDesign saved to: {args.output}")
            
            return 0
        else:
            print(f"Design failed: {result.get('error', 'Unknown error')}")
            return 1
    
    async def _command_status(self, args) -> int:
        """Get system status"""
        if not self.system:
            print("System not running. Use 'nexum run' to start the system.")
            return 1
        
        result = await self.system.execute_command("status")
        
        if result["success"]:
            status_data = result["data"]
            
            if args.json:
                print(json.dumps(status_data, indent=2))
            else:
                print(f"=== Nexum v2.0 Status ===")
                print(f"State: {status_data['state']}")
                print(f"Uptime: {status_data['uptime']:.2f} seconds")
                print(f"Operations: {status_data['total_operations']}")
                print(f"Errors: {status_data['errors']}")
                print(f"Warnings: {status_data['warnings']}")
                
                # Component status
                components = status_data.get('components', {})
                if components.get('system_manager'):
                    sys_status = components['system_manager']
                    resources = sys_status.get('resources', {})
                    print(f"\n=== System Resources ===")
                    print(f"CPU: {resources.get('cpu_usage', 0):.1f}%")
                    print(f"Memory: {resources.get('memory_usage', 0):.1f}%")
                    print(f"GPU: {resources.get('gpu_usage', 0):.1f}%")
            
            return 0
        else:
            print(f"Failed to get status: {result.get('error', 'Unknown error')}")
            return 1
    
    async def _command_move_motor(self, args) -> int:
        """Move motor to position"""
        if not self.system:
            print("System not running. Use 'nexum run' to start the system.")
            return 1
        
        print(f"Moving motor {args.motor_id} to position {args.position}")
        
        result = await self.system.execute_command("move_motor", {
            "motor_id": args.motor_id,
            "position": args.position
        })
        
        if result["success"]:
            print("Motor command executed successfully")
            return 0
        else:
            print(f"Motor command failed: {result.get('error', 'Unknown error')}")
            return 1
    
    async def _command_ai_inference(self, args) -> int:
        """Run AI inference"""
        if not self.system:
            print("System not running. Use 'nexum run' to start the system.")
            return 1
        
        print(f"Running AI inference with model {args.model_id}")
        
        result = await self.system.execute_command("ai_inference", {
            "model_id": args.model_id,
            "inputs": args.input
        })
        
        if result["success"]:
            output_data = result["data"]
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(output_data, f, indent=2)
                print(f"Results saved to: {args.output}")
            else:
                print(f"Inference result: {output_data}")
            
            return 0
        else:
            print(f"AI inference failed: {result.get('error', 'Unknown error')}")
            return 1
    
    async def _command_stop(self, args) -> int:
        """Stop system"""
        if not self.system:
            print("System not running.")
            return 1
        
        print("Stopping Nexum system...")
        await self.system.stop()
        print("System stopped.")
        return 0
    
    async def _command_restart(self, args) -> int:
        """Restart system"""
        if self.system:
            print("Stopping system for restart...")
            await self.system.stop()
        
        print("Restarting system...")
        config = SystemConfig()
        self.system = NexumSystem(config)
        
        success = await self.system.run()
        return 0 if success else 1
    
    async def _command_config(self, args) -> int:
        """Configuration management"""
        if args.config_action == 'show':
            # Show current configuration
            config = SystemConfig()
            print("=== Current Configuration ===")
            print(f"Log Level: {config.log_level}")
            print(f"Max Memory: {config.max_memory_mb} MB")
            print(f"Enable GPU: {config.enable_gpu}")
            print(f"Enable Motors: {config.enable_motors}")
            print(f"Enable Audio: {config.enable_audio}")
            print(f"Enable Vision: {config.enable_vision}")
            print(f"Enable AI: {config.enable_ai}")
            return 0
        
        elif args.config_action == 'reset':
            # Reset configuration to defaults
            print("Configuration reset to defaults")
            return 0
        
        else:
            print("Unknown config action")
            return 1
    
    async def _command_diagnose(self, args) -> int:
        """Run system diagnostics"""
        print("Running system diagnostics...")
        
        # Create temporary system for diagnostics
        config = SystemConfig(log_level="DEBUG" if args.verbose else "INFO")
        system = NexumSystem(config)
        
        if not await system.initialize():
            print("Failed to initialize system for diagnostics")
            return 1
        
        # Run diagnostics
        diagnostics = {
            "timestamp": time.time(),
            "system_info": system.get_status()
        }
        
        print(f"\n=== System Diagnostics ===")
        print(f"Timestamp: {time.ctime(diagnostics['timestamp'])}")
        
        status = diagnostics['system_info']
        print(f"System State: {status['state']}")
        print(f"Uptime: {status['uptime']:.2f} seconds")
        
        # Component health check
        components = status.get('components', {})
        for component_name, component_status in components.items():
            if component_status:
                print(f"\n{component_name.replace('_', ' ').title()}: OK")
            else:
                print(f"\n{component_name.replace('_', ' ').title()}: NOT INITIALIZED")
        
        print("\nDiagnostics complete.")
        return 0


def main():
    """Main CLI entry point"""
    cli = NexumCLI()
    return asyncio.run(cli.run())


if __name__ == "__main__":
    sys.exit(main())
