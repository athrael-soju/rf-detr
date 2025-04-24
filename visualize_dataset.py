#!/usr/bin/env python3
"""
Visualization tool for the entity tracking dataset.
This script generates various visualizations from the dataset JSON file.
"""

import argparse
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation
from collections import defaultdict
import pandas as pd
from datetime import datetime

def load_dataset(dataset_path):
    """Load dataset from JSON file."""
    with open(dataset_path, 'r') as f:
        return json.load(f)

def plot_entity_timeline(dataset, output_path=None):
    """Generate a timeline showing which entities are present in each frame."""
    print("Generating entity timeline...")
    
    # Find all unique entity IDs and frame IDs
    all_entity_ids = set()
    frame_ids = []
    
    for frame in dataset:
        frame_ids.append(frame['frame_id'])
        for entity in frame['entities']:
            all_entity_ids.add(entity['entity_id'])
    
    # For better visualization, group by entity type
    entity_by_type = defaultdict(list)
    for entity_id in sorted(all_entity_ids):
        # Extract entity type from ID (e.g., "person_0" -> "person")
        entity_type = entity_id.split('_')[0]
        entity_by_type[entity_type].append(entity_id)
    
    # Prepare data for visualization
    entity_presence = {}
    for entity_id in all_entity_ids:
        entity_presence[entity_id] = [False] * len(frame_ids)
    
    for i, frame in enumerate(dataset):
        entity_ids_in_frame = [entity['entity_id'] for entity in frame['entities']]
        for entity_id in entity_ids_in_frame:
            entity_presence[entity_id][i] = True
    
    # Create the visualization
    plt.figure(figsize=(12, max(8, len(all_entity_ids) * 0.25)))
    
    # Plot with colors by entity type
    colors = plt.cm.tab10(np.linspace(0, 1, len(entity_by_type)))
    color_map = {}
    
    y_ticks = []
    y_labels = []
    current_y = 0
    
    for i, (entity_type, entity_ids) in enumerate(entity_by_type.items()):
        for entity_id in entity_ids:
            plt.scatter(
                [frame_ids[j] for j in range(len(frame_ids)) if entity_presence[entity_id][j]], 
                [current_y] * sum(entity_presence[entity_id]), 
                c=[colors[i]], 
                s=10
            )
            # Draw connecting lines for consecutive frames
            consecutive_frames = []
            for j in range(len(frame_ids)):
                if entity_presence[entity_id][j]:
                    consecutive_frames.append(j)
                elif consecutive_frames:
                    if len(consecutive_frames) > 1:
                        plt.plot(
                            [frame_ids[k] for k in consecutive_frames], 
                            [current_y] * len(consecutive_frames), 
                            c=colors[i], 
                            alpha=0.6
                        )
                    consecutive_frames = []
            
            # Draw the last segment
            if consecutive_frames and len(consecutive_frames) > 1:
                plt.plot(
                    [frame_ids[k] for k in consecutive_frames], 
                    [current_y] * len(consecutive_frames), 
                    c=colors[i], 
                    alpha=0.6
                )
            
            y_ticks.append(current_y)
            y_labels.append(entity_id)
            current_y += 1
        
        # Add extra space between entity types
        current_y += 0.5
        
        # Save color for legend
        color_map[entity_type] = colors[i]
    
    # Create custom legend
    legend_elements = [
        patches.Patch(facecolor=color, label=entity_type)
        for entity_type, color in color_map.items()
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Set labels and ticks
    plt.yticks(y_ticks, y_labels)
    plt.xlabel('Frame ID')
    plt.ylabel('Entity ID')
    plt.title('Entity Presence Timeline')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Save the figure if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"Entity timeline saved to {output_path}")
    else:
        plt.tight_layout()
        plt.show()
    
    plt.close()

def plot_entity_changes(dataset, output_path=None):
    """Generate a visualization of entity changes between frames."""
    print("Generating entity changes visualization...")
    
    # Extract data
    frame_ids = []
    new_entities = []
    updated_entities = []
    removed_entities = []
    total_entities = []
    
    for frame in dataset:
        frame_ids.append(frame['frame_id'])
        new_entities.append(len(frame['delta']['new_entities']))
        updated_entities.append(len(frame['delta']['updated_entities']))
        removed_entities.append(len(frame['delta']['removed_entities']))
        total_entities.append(len(frame['entities']))
    
    # Create the visualization
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.bar(frame_ids, total_entities, color='blue', alpha=0.7, label='Total Entities')
    plt.xlabel('Frame ID')
    plt.ylabel('Count')
    plt.title('Total Entities per Frame')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.bar(frame_ids, new_entities, color='green', alpha=0.7, label='New')
    plt.bar(frame_ids, updated_entities, bottom=new_entities, color='orange', alpha=0.7, label='Updated')
    plt.bar(frame_ids, removed_entities, color='red', alpha=0.7, label='Removed')
    plt.xlabel('Frame ID')
    plt.ylabel('Count')
    plt.title('Entity Changes Between Frames')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save the figure if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"Entity changes visualization saved to {output_path}")
    else:
        plt.tight_layout()
        plt.show()
    
    plt.close()

def create_relationship_graph(frame, title):
    """Create a NetworkX graph for relationships in a frame."""
    G = nx.Graph()
    
    # Add nodes (entities)
    for entity in frame['entities']:
        G.add_node(
            entity['entity_id'], 
            type=entity['type'],
            confidence=entity['confidence']
        )
    
    # Add edges (relationships)
    for relationship in frame['relationships']:
        G.add_edge(
            relationship['subject'],
            relationship['object'],
            type=relationship['predicate']
        )
    
    return G

def plot_relationship_graph(dataset, frame_indices=None, output_dir=None):
    """Generate relationship graphs for specific frames."""
    print("Generating relationship graphs...")
    
    # If frame_indices not provided, choose evenly spaced frames
    if frame_indices is None:
        num_frames = len(dataset)
        if num_frames <= 4:
            frame_indices = list(range(num_frames))
        else:
            frame_indices = [0, num_frames // 3, 2 * num_frames // 3, num_frames - 1]
    
    # Create graphs for each selected frame
    for idx in frame_indices:
        if idx < 0 or idx >= len(dataset):
            print(f"Warning: Frame index {idx} out of range, skipping")
            continue
        
        frame = dataset[idx]
        title = f"Frame {frame['frame_id']} Relationships"
        G = create_relationship_graph(frame, title)
        
        if len(G.nodes) == 0:
            print(f"No relationships to visualize for frame {frame['frame_id']}")
            continue
        
        plt.figure(figsize=(10, 8))
        
        # Create node colors based on entity type
        entity_types = sorted(set(nx.get_node_attributes(G, 'type').values()))
        color_map = {t: plt.cm.tab10(i) for i, t in enumerate(entity_types)}
        node_colors = [color_map[G.nodes[node]['type']] for node in G.nodes]
        
        # Create node sizes based on confidence
        confidences = nx.get_node_attributes(G, 'confidence')
        node_sizes = [300 * conf for node, conf in confidences.items()]
        
        # Create edge colors based on relationship type
        edge_types = set()
        for _, _, data in G.edges(data=True):
            edge_types.add(data.get('type', 'unknown'))
        edge_types = sorted(edge_types)
        edge_color_map = {t: plt.cm.Set2(i) for i, t in enumerate(edge_types)}
        edge_colors = [edge_color_map[G.edges[edge]['type']] for edge in G.edges]
        
        # Draw the graph
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=2, alpha=0.7)
        nx.draw_networkx_labels(G, pos, font_size=8)
        
        # Create legends
        node_legend = [
            patches.Patch(color=color_map[t], label=t) 
            for t in entity_types
        ]
        edge_legend = [
            patches.Patch(color=edge_color_map[t], label=t) 
            for t in edge_types
        ]
        
        # Add legends in two columns
        plt.legend(handles=node_legend, title="Entity Types", loc='upper left')
        if edge_legend:
            plt.legend(handles=edge_legend, title="Relationship Types", loc='upper right')
        
        plt.title(title)
        plt.axis('off')
        
        # Save the figure if output_dir is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"relationships_frame_{frame['frame_id']}.png")
            plt.tight_layout()
            plt.savefig(output_path)
            print(f"Relationship graph saved to {output_path}")
        else:
            plt.tight_layout()
            plt.show()
        
        plt.close()

def animate_entity_movement(dataset, output_path=None, fps=5):
    """Create an animation showing entity movements across frames."""
    print("Generating entity movement animation...")
    
    # Extract all unique entity IDs and their positions over time
    entity_positions = defaultdict(list)
    frame_ids = []
    
    for frame in dataset:
        frame_ids.append(frame['frame_id'])
        entities_in_frame = {entity['entity_id']: entity for entity in frame['entities']}
        
        # For each entity, store position or None if not in this frame
        for entity_id in entity_positions.keys():
            if entity_id in entities_in_frame:
                entity = entities_in_frame[entity_id]
                bbox = entity['bbox']
                # Use center of bounding box
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                entity_positions[entity_id].append((center_x, center_y))
            else:
                entity_positions[entity_id].append(None)
        
        # Add new entities
        for entity_id, entity in entities_in_frame.items():
            if entity_id not in entity_positions:
                # Pad with None for previous frames
                entity_positions[entity_id] = [None] * (len(frame_ids) - 1)
                # Add position for current frame
                bbox = entity['bbox']
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                entity_positions[entity_id].append((center_x, center_y))
    
    # Find max frame dimensions
    max_width = 0
    max_height = 0
    for frame in dataset:
        for entity in frame['entities']:
            bbox = entity['bbox']
            max_width = max(max_width, bbox[2])
            max_height = max(max_height, bbox[3])
    
    # Create animation
    fig, ax = plt.subplots(figsize=(10, max_height/max_width * 10))
    
    # Create color map based on entity type
    entity_types = {}
    for frame in dataset:
        for entity in frame['entities']:
            entity_id = entity['entity_id']
            entity_type = entity['type']
            entity_types[entity_id] = entity_type
    
    unique_types = sorted(set(entity_types.values()))
    color_map = {t: plt.cm.tab10(i) for i, t in enumerate(unique_types)}
    
    def init():
        ax.clear()
        ax.set_xlim(0, max_width)
        ax.set_ylim(max_height, 0)  # Invert y-axis for image coordinates
        ax.set_title('Entity Movement Animation (Frame 0)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        return []
    
    def update(frame_idx):
        ax.clear()
        
        # Set plot limits and labels
        ax.set_xlim(0, max_width)
        ax.set_ylim(max_height, 0)  # Invert y-axis for image coordinates
        ax.set_title(f'Entity Movement Animation (Frame {frame_ids[frame_idx]})')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # Draw entities in this frame
        for entity_id, positions in entity_positions.items():
            if frame_idx < len(positions) and positions[frame_idx] is not None:
                x, y = positions[frame_idx]
                entity_type = entity_types.get(entity_id, 'unknown')
                color = color_map.get(entity_type, 'gray')
                
                # Draw point and label
                ax.scatter(x, y, color=color, s=50, label=entity_type)
                ax.text(x + 10, y, entity_id, fontsize=8)
                
                # Draw trail for previous positions
                trail_positions = [(p[0], p[1]) for p in positions[:frame_idx] if p is not None]
                if trail_positions:
                    trail_x, trail_y = zip(*trail_positions)
                    ax.plot(trail_x, trail_y, color=color, alpha=0.3, linewidth=1)
        
        # Create legend with unique entity types
        handles = [
            patches.Patch(color=color_map[t], label=t)
            for t in unique_types
        ]
        ax.legend(handles=handles, loc='upper right')
        
        return []
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=len(frame_ids), init_func=init, blit=True)
    
    # Save animation or display it
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        ani.save(output_path, writer='pillow', fps=fps)
        print(f"Animation saved to {output_path}")
    else:
        plt.tight_layout()
        plt.show()
    
    plt.close()

def generate_entity_statistics(dataset, output_path=None):
    """Generate statistics about entities in the dataset."""
    print("Generating entity statistics...")
    
    # Extract data for analysis
    entity_types = defaultdict(int)
    entity_lifespans = defaultdict(int)
    entity_first_appearance = {}
    relationship_types = defaultdict(int)
    
    # Track entity appearances
    entity_frames = defaultdict(set)
    
    for frame in dataset:
        frame_id = frame['frame_id']
        
        # Count entity types and record appearances
        for entity in frame['entities']:
            entity_id = entity['entity_id']
            entity_type = entity['type']
            
            entity_types[entity_type] += 1
            entity_frames[entity_id].add(frame_id)
            
            if entity_id not in entity_first_appearance:
                entity_first_appearance[entity_id] = frame_id
        
        # Count relationship types
        for relationship in frame['relationships']:
            relationship_types[relationship['predicate']] += 1
    
    # Calculate entity lifespans
    for entity_id, frames in entity_frames.items():
        entity_lifespans[entity_id] = max(frames) - min(frames) + 1
    
    # Calculate average lifespan by entity type
    type_lifespans = defaultdict(list)
    for entity_id, lifespan in entity_lifespans.items():
        # Find entity type from the ID
        entity_type = entity_id.split('_')[0]
        type_lifespans[entity_type].append(lifespan)
    
    avg_lifespans = {
        entity_type: sum(lifespans) / len(lifespans)
        for entity_type, lifespans in type_lifespans.items()
    }
    
    # Generate statistics table
    entity_stats = pd.DataFrame({
        'Entity Type': list(entity_types.keys()),
        'Count': list(entity_types.values()),
        'Avg Lifespan (frames)': [avg_lifespans.get(t, 0) for t in entity_types.keys()]
    })
    
    relationship_stats = pd.DataFrame({
        'Relationship Type': list(relationship_types.keys()),
        'Count': list(relationship_types.values())
    })
    
    # Create visualizations
    plt.figure(figsize=(12, 12))
    
    # Entity type distribution
    plt.subplot(2, 2, 1)
    plt.bar(entity_stats['Entity Type'], entity_stats['Count'], color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.title('Entity Type Distribution')
    plt.ylabel('Count')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Average lifespan by entity type
    plt.subplot(2, 2, 2)
    plt.bar(entity_stats['Entity Type'], entity_stats['Avg Lifespan (frames)'], color='lightgreen')
    plt.xticks(rotation=45, ha='right')
    plt.title('Average Entity Lifespan by Type')
    plt.ylabel('Frames')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Relationship type distribution
    plt.subplot(2, 2, 3)
    plt.bar(relationship_stats['Relationship Type'], relationship_stats['Count'], color='salmon')
    plt.xticks(rotation=45, ha='right')
    plt.title('Relationship Type Distribution')
    plt.ylabel('Count')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Entity lifespan histogram
    plt.subplot(2, 2, 4)
    plt.hist(list(entity_lifespans.values()), bins=10, color='mediumpurple', alpha=0.7)
    plt.title('Entity Lifespan Distribution')
    plt.xlabel('Lifespan (frames)')
    plt.ylabel('Number of Entities')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save or display the figure
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"Entity statistics saved to {output_path}")
    else:
        plt.tight_layout()
        plt.show()
    
    plt.close()
    
    # Print summary statistics
    print("\nEntity Statistics Summary:")
    print(entity_stats)
    print("\nRelationship Statistics Summary:")
    print(relationship_stats)
    
    return entity_stats, relationship_stats

def main():
    parser = argparse.ArgumentParser(description="Visualize entity tracking dataset")
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset JSON file')
    parser.add_argument('--output-dir', type=str, default='./output/visualizations', help='Directory to save visualizations')
    parser.add_argument('--timeline', action='store_true', help='Generate entity timeline visualization')
    parser.add_argument('--changes', action='store_true', help='Generate entity changes visualization')
    parser.add_argument('--relationships', action='store_true', help='Generate relationship graph visualizations')
    parser.add_argument('--animation', action='store_true', help='Generate entity movement animation')
    parser.add_argument('--statistics', action='store_true', help='Generate entity statistics')
    parser.add_argument('--all', action='store_true', help='Generate all visualizations')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    dataset = load_dataset(args.dataset)
    print(f"Loaded dataset with {len(dataset)} frames")
    
    # Determine output name base from dataset filename
    dataset_basename = os.path.splitext(os.path.basename(args.dataset))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate requested visualizations
    if args.all or args.timeline:
        output_path = os.path.join(args.output_dir, f"{dataset_basename}_{timestamp}_timeline.png")
        plot_entity_timeline(dataset, output_path)
    
    if args.all or args.changes:
        output_path = os.path.join(args.output_dir, f"{dataset_basename}_{timestamp}_changes.png")
        plot_entity_changes(dataset, output_path)
    
    if args.all or args.relationships:
        relationship_dir = os.path.join(args.output_dir, f"{dataset_basename}_{timestamp}_relationships")
        plot_relationship_graph(dataset, output_dir=relationship_dir)
    
    if args.all or args.animation:
        output_path = os.path.join(args.output_dir, f"{dataset_basename}_{timestamp}_animation.gif")
        animate_entity_movement(dataset, output_path)
    
    if args.all or args.statistics:
        output_path = os.path.join(args.output_dir, f"{dataset_basename}_{timestamp}_statistics.png")
        generate_entity_statistics(dataset, output_path)
    
    if not (args.all or args.timeline or args.changes or args.relationships or args.animation or args.statistics):
        print("No visualizations selected. Use --all or specify the visualizations to generate.")
        print("Available options: --timeline, --changes, --relationships, --animation, --statistics")

if __name__ == "__main__":
    main() 