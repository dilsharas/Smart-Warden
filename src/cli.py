#!/usr/bin/env python3
"""
Command Line Interface for Smart Contract AI Analyzer

This CLI provides access to all major functionality of the Smart Contract AI Analyzer,
including vulnerability detection, tool comparison, and batch processing.
"""

import argparse
import json
import sys
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

# Import core components
from src.features.feature_extractor import FeatureExtractor
from src.models.random_forest import RandomForestModel
from src.models.multiclass_classifier import MultiClassClassifier
from src.integration.slither_runner import SlitherRunner
from src.integration.mythril_runner import MythrilRunner
from src.integration.tool_comparator import ToolComparator
from src.utils.logging_config import setup_logging

# Version information
__version__ = "1.0.0"

class SmartContractAnalyzerCLI:
    """Main CLI class for Smart Contract AI Analyzer"""
    
    def __init__(self):
        self.feature_extractor = None
        self.binary_model = None
        self.multiclass_model = None
        self.slither_runner = None
        self.mythril_runner = None
        self.tool_comparator = None
        self.logger = None
        
    def setup_logging(self, verbose: bool = False):
        """Setup logging configuration"""
        level = logging.DEBUG if verbose else logging.INFO
        self.logger = setup_logging(level=level)
        
    def initialize_components(self, config_path: Optional[str] = None):
        """Initialize all analysis components"""
        try:
            self.logger.info("Initializing Smart Contract AI Analyzer components...")
            
            # Initialize feature extractor
            self.feature_extractor = FeatureExtractor()
            self.logger.info("✓ Feature extractor initialized")
            
            # Initialize models
            self.binary_model = RandomForestModel()
            self.multiclass_model = MultiClassClassifier()
            
            # Load pre-trained models if available
            binary_model_path = Path("models/binary_classifier.joblib")
            multiclass_model_path = Path("models/multiclass_classifier.joblib")
            
            if binary_model_path.exists():
                self.binary_model.load_model(str(binary_model_path))
                self.logger.info("✓ Binary classifier loaded")
            else:
                self.logger.warning("⚠ Binary classifier model not found")
                
            if multiclass_model_path.exists():
                self.multiclass_model.load_model(str(multiclass_model_path))
                self.logger.info("✓ Multi-class classifier loaded")
            else:
                self.logger.warning("⚠ Multi-class classifier model not found")
            
            # Initialize external tools
            self.slither_runner = SlitherRunner()
            self.mythril_runner = MythrilRunner()
            self.tool_comparator = ToolComparator()
            self.logger.info("✓ External tools initialized")
            
            self.logger.info("All components initialized successfully!")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            sys.exit(1)
    
    def analyze_contract(self, contract_path: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single smart contract"""
        try:
            # Read contract code
            with open(contract_path, 'r', encoding='utf-8') as f:
                contract_code = f.read()
            
            self.logger.info(f"Analyzing contract: {contract_path}")
            
            results = {
                'contract_path': contract_path,
                'timestamp': datetime.now().isoformat(),
                'analysis_options': options,
                'results': {}
            }
            
            # AI Analysis
            if options.get('ai_analysis', True):
                self.logger.info("Running AI analysis...")
                start_time = time.time()
                
                # Extract features
                features = self.feature_extractor.extract_features(contract_code)
                
                # Binary classification
                if self.binary_model and hasattr(self.binary_model, 'model') and self.binary_model.model:
                    binary_pred = self.binary_model.predict([features])
                    binary_prob = self.binary_model.predict_proba([features])
                    
                    results['results']['ai_analysis'] = {
                        'binary_classification': {
                            'prediction': 'vulnerable' if binary_pred[0] == 1 else 'safe',
                            'confidence': float(max(binary_prob[0])),
                            'vulnerability_probability': float(binary_prob[0][1])
                        }
                    }
                
                # Multi-class classification
                if self.multiclass_model and hasattr(self.multiclass_model, 'model') and self.multiclass_model.model:
                    multiclass_pred = self.multiclass_model.predict([features])
                    multiclass_prob = self.multiclass_model.predict_proba([features])
                    
                    vulnerability_types = ['safe', 'reentrancy', 'access_control', 'arithmetic', 'unchecked_calls', 'dos', 'bad_randomness']
                    
                    results['results']['ai_analysis']['multiclass_classification'] = {
                        'predicted_type': vulnerability_types[multiclass_pred[0]],
                        'probabilities': {
                            vuln_type: float(prob) 
                            for vuln_type, prob in zip(vulnerability_types, multiclass_prob[0])
                        }
                    }
                
                ai_time = time.time() - start_time
                results['results']['ai_analysis']['execution_time'] = ai_time
                self.logger.info(f"✓ AI analysis completed in {ai_time:.2f}s")
            
            # Slither Analysis
            if options.get('slither_analysis', False):
                self.logger.info("Running Slither analysis...")
                start_time = time.time()
                
                slither_results = self.slither_runner.analyze_contract(contract_code)
                slither_time = time.time() - start_time
                
                results['results']['slither_analysis'] = slither_results
                results['results']['slither_analysis']['execution_time'] = slither_time
                self.logger.info(f"✓ Slither analysis completed in {slither_time:.2f}s")
            
            # Mythril Analysis
            if options.get('mythril_analysis', False):
                self.logger.info("Running Mythril analysis...")
                start_time = time.time()
                
                mythril_results = self.mythril_runner.analyze_contract(contract_code)
                mythril_time = time.time() - start_time
                
                results['results']['mythril_analysis'] = mythril_results
                results['results']['mythril_analysis']['execution_time'] = mythril_time
                self.logger.info(f"✓ Mythril analysis completed in {mythril_time:.2f}s")
            
            # Tool Comparison
            if options.get('compare_tools', False) and len(results['results']) > 1:
                self.logger.info("Comparing tool results...")
                comparison = self.tool_comparator.compare_results(results['results'])
                results['results']['tool_comparison'] = comparison
                self.logger.info("✓ Tool comparison completed")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing contract {contract_path}: {e}")
            return {
                'contract_path': contract_path,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def batch_analyze(self, input_dir: str, output_dir: str, options: Dict[str, Any]):
        """Analyze multiple contracts in batch"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        if not input_path.exists():
            self.logger.error(f"Input directory does not exist: {input_dir}")
            return
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all Solidity files
        sol_files = list(input_path.rglob("*.sol"))
        
        if not sol_files:
            self.logger.warning(f"No .sol files found in {input_dir}")
            return
        
        self.logger.info(f"Found {len(sol_files)} contracts to analyze")
        
        results = []
        failed_analyses = []
        
        for i, contract_file in enumerate(sol_files, 1):
            self.logger.info(f"Processing {i}/{len(sol_files)}: {contract_file.name}")
            
            try:
                result = self.analyze_contract(str(contract_file), options)
                results.append(result)
                
                # Save individual result
                result_file = output_path / f"{contract_file.stem}_analysis.json"
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2)
                    
            except Exception as e:
                self.logger.error(f"Failed to analyze {contract_file}: {e}")
                failed_analyses.append(str(contract_file))
        
        # Save batch summary
        summary = {
            'batch_analysis_summary': {
                'total_contracts': len(sol_files),
                'successful_analyses': len(results),
                'failed_analyses': len(failed_analyses),
                'failed_files': failed_analyses,
                'timestamp': datetime.now().isoformat(),
                'options': options
            },
            'results': results
        }
        
        summary_file = output_path / "batch_analysis_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Batch analysis completed. Results saved to {output_path}")
        self.logger.info(f"Successfully analyzed: {len(results)}/{len(sol_files)} contracts")
    
    def generate_report(self, results_file: str, output_format: str = 'json'):
        """Generate analysis report in specified format"""
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            if output_format == 'json':
                # Pretty print JSON
                print(json.dumps(results, indent=2))
            
            elif output_format == 'summary':
                # Generate text summary
                self._print_summary_report(results)
            
            elif output_format == 'csv':
                # Generate CSV report
                self._generate_csv_report(results)
            
            else:
                self.logger.error(f"Unsupported output format: {output_format}")
                
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
    
    def _print_summary_report(self, results: Dict[str, Any]):
        """Print a human-readable summary report"""
        print("\n" + "="*60)
        print("SMART CONTRACT ANALYSIS REPORT")
        print("="*60)
        
        if 'batch_analysis_summary' in results:
            # Batch analysis summary
            summary = results['batch_analysis_summary']
            print(f"\nBatch Analysis Summary:")
            print(f"  Total Contracts: {summary['total_contracts']}")
            print(f"  Successful: {summary['successful_analyses']}")
            print(f"  Failed: {summary['failed_analyses']}")
            print(f"  Timestamp: {summary['timestamp']}")
            
            # Aggregate statistics
            vulnerable_count = 0
            safe_count = 0
            
            for result in results.get('results', []):
                if 'results' in result and 'ai_analysis' in result['results']:
                    ai_result = result['results']['ai_analysis']
                    if 'binary_classification' in ai_result:
                        if ai_result['binary_classification']['prediction'] == 'vulnerable':
                            vulnerable_count += 1
                        else:
                            safe_count += 1
            
            print(f"\nVulnerability Summary:")
            print(f"  Vulnerable Contracts: {vulnerable_count}")
            print(f"  Safe Contracts: {safe_count}")
            
        else:
            # Single contract analysis
            print(f"\nContract: {results.get('contract_path', 'Unknown')}")
            print(f"Timestamp: {results.get('timestamp', 'Unknown')}")
            
            if 'results' in results:
                analysis_results = results['results']
                
                # AI Analysis results
                if 'ai_analysis' in analysis_results:
                    ai_result = analysis_results['ai_analysis']
                    print(f"\nAI Analysis:")
                    
                    if 'binary_classification' in ai_result:
                        binary = ai_result['binary_classification']
                        print(f"  Prediction: {binary['prediction'].upper()}")
                        print(f"  Confidence: {binary['confidence']:.2%}")
                        print(f"  Vulnerability Probability: {binary['vulnerability_probability']:.2%}")
                    
                    if 'multiclass_classification' in ai_result:
                        multiclass = ai_result['multiclass_classification']
                        print(f"  Predicted Type: {multiclass['predicted_type']}")
                        print(f"  Top Probabilities:")
                        sorted_probs = sorted(multiclass['probabilities'].items(), 
                                            key=lambda x: x[1], reverse=True)
                        for vuln_type, prob in sorted_probs[:3]:
                            print(f"    {vuln_type}: {prob:.2%}")
                
                # External tool results
                for tool in ['slither_analysis', 'mythril_analysis']:
                    if tool in analysis_results:
                        tool_result = analysis_results[tool]
                        tool_name = tool.replace('_analysis', '').title()
                        print(f"\n{tool_name} Analysis:")
                        
                        if 'vulnerabilities' in tool_result:
                            vulns = tool_result['vulnerabilities']
                            print(f"  Vulnerabilities Found: {len(vulns)}")
                            for vuln in vulns[:3]:  # Show top 3
                                print(f"    - {vuln.get('type', 'Unknown')}: {vuln.get('severity', 'Unknown')}")
                        
                        if 'execution_time' in tool_result:
                            print(f"  Execution Time: {tool_result['execution_time']:.2f}s")
        
        print("\n" + "="*60)
    
    def _generate_csv_report(self, results: Dict[str, Any]):
        """Generate CSV report"""
        import csv
        import io
        
        output = io.StringIO()
        
        if 'batch_analysis_summary' in results:
            # Batch CSV
            fieldnames = ['contract_path', 'ai_prediction', 'ai_confidence', 'predicted_type', 
                         'slither_vulnerabilities', 'mythril_vulnerabilities', 'analysis_time']
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results.get('results', []):
                row = {'contract_path': result.get('contract_path', '')}
                
                if 'results' in result:
                    analysis_results = result['results']
                    
                    # AI results
                    if 'ai_analysis' in analysis_results:
                        ai_result = analysis_results['ai_analysis']
                        if 'binary_classification' in ai_result:
                            binary = ai_result['binary_classification']
                            row['ai_prediction'] = binary['prediction']
                            row['ai_confidence'] = binary['confidence']
                        
                        if 'multiclass_classification' in ai_result:
                            multiclass = ai_result['multiclass_classification']
                            row['predicted_type'] = multiclass['predicted_type']
                    
                    # Tool results
                    if 'slither_analysis' in analysis_results:
                        slither_vulns = analysis_results['slither_analysis'].get('vulnerabilities', [])
                        row['slither_vulnerabilities'] = len(slither_vulns)
                    
                    if 'mythril_analysis' in analysis_results:
                        mythril_vulns = analysis_results['mythril_analysis'].get('vulnerabilities', [])
                        row['mythril_vulnerabilities'] = len(mythril_vulns)
                
                writer.writerow(row)
        
        print(output.getvalue())
    
    def list_models(self):
        """List available models"""
        models_dir = Path("models")
        
        print("\nAvailable Models:")
        print("-" * 40)
        
        if models_dir.exists():
            model_files = list(models_dir.glob("*.joblib"))
            if model_files:
                for model_file in model_files:
                    size = model_file.stat().st_size / (1024 * 1024)  # MB
                    modified = datetime.fromtimestamp(model_file.stat().st_mtime)
                    print(f"  {model_file.name}")
                    print(f"    Size: {size:.1f} MB")
                    print(f"    Modified: {modified.strftime('%Y-%m-%d %H:%M:%S')}")
                    print()
            else:
                print("  No model files found in models/ directory")
        else:
            print("  Models directory not found")
    
    def check_tools(self):
        """Check availability of external tools"""
        print("\nExternal Tools Status:")
        print("-" * 40)
        
        # Check Slither
        try:
            slither_runner = SlitherRunner()
            if slither_runner.check_installation():
                print("  ✓ Slither: Available")
            else:
                print("  ✗ Slither: Not available")
        except Exception as e:
            print(f"  ✗ Slither: Error - {e}")
        
        # Check Mythril
        try:
            mythril_runner = MythrilRunner()
            if mythril_runner.check_installation():
                print("  ✓ Mythril: Available")
            else:
                print("  ✗ Mythril: Not available")
        except Exception as e:
            print(f"  ✗ Mythril: Error - {e}")


def create_parser():
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description="Smart Contract AI Analyzer - CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single contract with AI only
  python -m src.cli analyze contract.sol
  
  # Analyze with all tools
  python -m src.cli analyze contract.sol --slither --mythril --compare
  
  # Batch analysis
  python -m src.cli batch-analyze contracts/ results/ --slither
  
  # Generate report
  python -m src.cli report results.json --format summary
  
  # Check system status
  python -m src.cli status
        """
    )
    
    parser.add_argument('--version', action='version', version=f'Smart Contract AI Analyzer {__version__}')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--config', help='Path to configuration file')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a single smart contract')
    analyze_parser.add_argument('contract', help='Path to smart contract file')
    analyze_parser.add_argument('--output', '-o', help='Output file for results (JSON format)')
    analyze_parser.add_argument('--no-ai', action='store_true', help='Disable AI analysis')
    analyze_parser.add_argument('--slither', action='store_true', help='Enable Slither analysis')
    analyze_parser.add_argument('--mythril', action='store_true', help='Enable Mythril analysis')
    analyze_parser.add_argument('--compare', action='store_true', help='Compare tool results')
    analyze_parser.add_argument('--timeout', type=int, default=300, help='Analysis timeout in seconds')
    
    # Batch analyze command
    batch_parser = subparsers.add_parser('batch-analyze', help='Analyze multiple contracts')
    batch_parser.add_argument('input_dir', help='Directory containing smart contracts')
    batch_parser.add_argument('output_dir', help='Directory to save results')
    batch_parser.add_argument('--no-ai', action='store_true', help='Disable AI analysis')
    batch_parser.add_argument('--slither', action='store_true', help='Enable Slither analysis')
    batch_parser.add_argument('--mythril', action='store_true', help='Enable Mythril analysis')
    batch_parser.add_argument('--compare', action='store_true', help='Compare tool results')
    batch_parser.add_argument('--parallel', type=int, default=1, help='Number of parallel processes')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate analysis report')
    report_parser.add_argument('results_file', help='Path to results JSON file')
    report_parser.add_argument('--format', choices=['json', 'summary', 'csv'], 
                              default='summary', help='Output format')
    report_parser.add_argument('--output', '-o', help='Output file (default: stdout)')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Check system status')
    
    # Models command
    models_parser = subparsers.add_parser('models', help='List available models')
    
    return parser


def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize CLI
    cli = SmartContractAnalyzerCLI()
    cli.setup_logging(args.verbose)
    
    try:
        if args.command in ['analyze', 'batch-analyze']:
            cli.initialize_components(args.config)
        
        # Execute commands
        if args.command == 'analyze':
            options = {
                'ai_analysis': not args.no_ai,
                'slither_analysis': args.slither,
                'mythril_analysis': args.mythril,
                'compare_tools': args.compare,
                'timeout': args.timeout
            }
            
            result = cli.analyze_contract(args.contract, options)
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2)
                cli.logger.info(f"Results saved to {args.output}")
            else:
                print(json.dumps(result, indent=2))
        
        elif args.command == 'batch-analyze':
            options = {
                'ai_analysis': not args.no_ai,
                'slither_analysis': args.slither,
                'mythril_analysis': args.mythril,
                'compare_tools': args.compare
            }
            
            cli.batch_analyze(args.input_dir, args.output_dir, options)
        
        elif args.command == 'report':
            if args.output:
                # Redirect stdout to file
                import sys
                with open(args.output, 'w') as f:
                    old_stdout = sys.stdout
                    sys.stdout = f
                    cli.generate_report(args.results_file, args.format)
                    sys.stdout = old_stdout
                cli.logger.info(f"Report saved to {args.output}")
            else:
                cli.generate_report(args.results_file, args.format)
        
        elif args.command == 'status':
            cli.check_tools()
        
        elif args.command == 'models':
            cli.list_models()
    
    except KeyboardInterrupt:
        cli.logger.info("Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        cli.logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()