# main.py
import argparse
import sys
import os
import json

from src.core.graph_analyzer import GraphAnalyzer
from src.core.content_moderator import ContentModerator
from src.core.influence_detector import InfluenceDetector
from src.core.network_dynamics import NetworkDynamics
from src.utils.data_processor import DataProcessor
from src.utils.visualization import NetworkVisualizer

def main():
    parser = argparse.ArgumentParser(description="SocialSentinel - Network Analysis Platform")
    
    parser.add_argument("--analyze-network", type=str, help="Analyze network from edge file")
    parser.add_argument("--moderate-content", type=str, help="Moderate content from text file")
    parser.add_argument("--detect-influence", type=str, help="Detect influence in network")
    parser.add_argument("--platform", type=str, default="twitter", help="Social media platform")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    
    args = parser.parse_args()
    
    data_processor = DataProcessor()
    
    if args.analyze_network:
        try:
            processed_data = data_processor.load_social_media_data(args.analyze_network, args.platform)
            
            graph_analyzer = GraphAnalyzer()
            graph_analyzer.build_graph_from_edges(processed_data['edges'])
            
            if processed_data['node_features']:
                graph_analyzer.add_node_features(processed_data['node_features'])
            
            centrality = graph_analyzer.compute_centrality()
            communities = graph_analyzer.detect_communities()
            statistics = graph_analyzer.get_graph_statistics()
            
            print("Network Analysis Results:")
            print(f"Nodes: {statistics['num_nodes']}, Edges: {statistics['num_edges']}")
            print(f"Density: {statistics['density']:.4f}")
            print(f"Communities: {len(set(communities.values()))}")
            
            if args.visualize:
                visualizer = NetworkVisualizer()
                fig = visualizer.plot_community_structure(graph_analyzer.graph, communities)
                if args.output:
                    fig.savefig(f"{args.output}_network.png")
                    print(f"Visualization saved to {args.output}_network.png")
            
            if args.output:
                results = {
                    'centrality': centrality,
                    'communities': communities,
                    'statistics': statistics
                }
                with open(f"{args.output}_analysis.json", 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Results saved to {args.output}_analysis.json")
        
        except Exception as e:
            print(f"Error analyzing network: {e}")
    
    elif args.moderate_content:
        try:
            with open(args.moderate_content, 'r') as f:
                texts = [line.strip() for line in f if line.strip()]
            
            moderator = ContentModerator()
            analyses = moderator.batch_analyze(texts)
            metrics = moderator.get_content_metrics(texts)
            
            print("Content Moderation Results:")
            print(f"Total texts: {metrics['total_texts']}")
            print(f"Harmful content: {metrics['harmful_content_count']} ({metrics['harmful_percentage']:.1f}%)")
            print(f"Pattern distribution: {metrics['pattern_distribution']}")
            
            harmful_texts = [text for i, text in enumerate(texts) if analyses[i]['is_harmful']]
            if harmful_texts:
                print("\nHarmful texts detected:")
                for text in harmful_texts[:5]:
                    print(f"  - {text[:100]}...")
            
            if args.output:
                results = {
                    'analyses': analyses,
                    'metrics': metrics
                }
                with open(f"{args.output}_moderation.json", 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Results saved to {args.output}_moderation.json")
        
        except Exception as e:
            print(f"Error moderating content: {e}")
    
    elif args.detect_influence:
        try:
            processed_data = data_processor.load_social_media_data(args.detect_influence, args.platform)
            
            graph_analyzer = GraphAnalyzer()
            graph_analyzer.build_graph_from_edges(processed_data['edges'])
            
            influence_detector = InfluenceDetector()
            influence_scores = influence_detector.calculate_influence_metrics(graph_analyzer.graph)
            
            top_influencers = sorted(influence_scores.items(), key=lambda x: x[1], reverse=True)[:10]
            
            print("Top Influencers:")
            for node, score in top_influencers:
                print(f"  Node {node}: {score:.4f}")
            
            if args.visualize:
                visualizer = NetworkVisualizer()
                fig = visualizer.plot_influence_distribution(influence_scores)
                if args.output:
                    fig.savefig(f"{args.output}_influence.png")
                    print(f"Visualization saved to {args.output}_influence.png")
            
            if args.output:
                results = {
                    'influence_scores': influence_scores,
                    'top_influencers': top_influencers
                }
                with open(f"{args.output}_influence.json", 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Results saved to {args.output}_influence.json")
        
        except Exception as e:
            print(f"Error detecting influence: {e}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()