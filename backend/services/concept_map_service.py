"""
Concept Map Service for Visual Learning.

This module generates visual concept maps from educational content
using graph algorithms and exports to SVG/PNG formats.
"""
import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import json
import re
from collections import defaultdict

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Concept map node types."""
    MAIN_CONCEPT = "main_concept"
    SUB_CONCEPT = "sub_concept"
    SUPPORTING_DETAIL = "supporting_detail"
    EXAMPLE = "example"
    DEFINITION = "definition"


class EdgeType(Enum):
    """Concept map edge types."""
    IS_A = "is_a"
    HAS = "has"
    LEADS_TO = "leads_to"
    EXAMPLE_OF = "example_of"
    CAUSES = "causes"
    REQUIRES = "requires"
    RELATED_TO = "related_to"


@dataclass
class ConceptNode:
    """Node in concept map."""
    id: str
    label: str
    type: NodeType
    level: int  # hierarchy level (0 = root)
    importance: float  # 0-1
    metadata: Dict[str, Any]


@dataclass
class ConceptEdge:
    """Edge connecting concepts."""
    source_id: str
    target_id: str
    edge_type: EdgeType
    label: str
    weight: float  # 0-1


@dataclass
class ConceptMap:
    """Complete concept map structure."""
    title: str
    subject: str
    grade: int
    nodes: List[ConceptNode]
    edges: List[ConceptEdge]
    root_node_id: str
    metadata: Dict[str, Any]


class ConceptExtractor:
    """
    Extracts concepts from educational content.
    """
    
    # Keywords for identifying concept relationships
    RELATIONSHIP_KEYWORDS = {
        EdgeType.IS_A: ['is', 'are', 'is a', 'are a', 'is an', 'are an'],
        EdgeType.HAS: ['has', 'have', 'contains', 'includes', 'consists of'],
        EdgeType.LEADS_TO: ['leads to', 'results in', 'causes', 'produces'],
        EdgeType.CAUSES: ['causes', 'results in', 'leads to', 'produces', 'creates'],
        EdgeType.REQUIRES: ['requires', 'needs', 'depends on', 'relies on'],
        EdgeType.EXAMPLE_OF: ['for example', 'such as', 'like', 'e.g.', 'instance of'],
    }
    
    @classmethod
    def extract_concepts(cls, content: str, subject: str) -> List[str]:
        """
        Extract key concepts from content.
        
        Args:
            content: Educational content text
            subject: Subject area
        
        Returns:
            List of extracted concepts
        """
        concepts = []
        
        # Extract noun phrases (simplified approach)
        # In production, use NLP library like spaCy
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', content)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Extract capitalized terms (likely concepts)
            capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', sentence)
            concepts.extend(capitalized)
            
            # Extract terms in bold (if markdown)
            bold = re.findall(r'\*\*([^*]+)\*\*', sentence)
            concepts.extend(bold)
            
            # Subject-specific keywords
            if subject.lower() == 'science':
                science_keywords = cls._extract_science_concepts(sentence)
                concepts.extend(science_keywords)
            elif subject.lower() == 'math':
                math_keywords = cls._extract_math_concepts(sentence)
                concepts.extend(math_keywords)
        
        # Remove duplicates and clean
        concepts = list({c.strip() for c in concepts if c.strip()})
        
        logger.debug(f"Extracted {len(concepts)} concepts")
        return concepts
    
    @classmethod
    def _extract_science_concepts(cls, text: str) -> List[str]:
        """Extract science-specific concepts."""
        keywords = [
            'photosynthesis', 'respiration', 'cell', 'nucleus', 'mitochondria',
            'chloroplast', 'energy', 'glucose', 'oxygen', 'carbon dioxide',
            'enzyme', 'catalyst', 'reaction', 'molecule', 'atom', 'electron',
            'force', 'gravity', 'velocity', 'acceleration', 'momentum'
        ]
        
        found = []
        text_lower = text.lower()
        for keyword in keywords:
            if keyword in text_lower:
                found.append(keyword.capitalize())
        
        return found
    
    @classmethod
    def _extract_math_concepts(cls, text: str) -> List[str]:
        """Extract math-specific concepts."""
        keywords = [
            'equation', 'variable', 'constant', 'function', 'derivative',
            'integral', 'matrix', 'vector', 'polynomial', 'theorem',
            'axiom', 'proof', 'triangle', 'circle', 'angle', 'area',
            'perimeter', 'volume', 'fraction', 'ratio', 'proportion'
        ]
        
        found = []
        text_lower = text.lower()
        for keyword in keywords:
            if keyword in text_lower:
                found.append(keyword.capitalize())
        
        return found
    
    @classmethod
    @classmethod
    def _find_concepts_in_sentence_parts(
        cls,
        parts: List[str],
        concepts_in_sentence: List[str]
    ) -> Tuple[List[str], List[str]]:
        """Find concepts before and after a relationship keyword."""
        if len(parts) != 2:
            return [], []
        
        before_concepts = [c for c in concepts_in_sentence if c.lower() in parts[0]]
        after_concepts = [c for c in concepts_in_sentence if c.lower() in parts[1]]
        
        return before_concepts, after_concepts
    
    @classmethod
    def _extract_relationship_from_sentence(
        cls,
        sentence_lower: str,
        concepts_in_sentence: List[str]
    ) -> List[Tuple[str, str, EdgeType]]:
        """Extract relationships from a single sentence."""
        relationships = []
        
        for edge_type, keywords in cls.RELATIONSHIP_KEYWORDS.items():
            for keyword in keywords:
                if keyword in sentence_lower:
                    parts = sentence_lower.split(keyword, 1)
                    before, after = cls._find_concepts_in_sentence_parts(parts, concepts_in_sentence)
                    
                    if before and after:
                        relationships.append((before[-1], after[0], edge_type))
        
        return relationships
    
    @classmethod
    def identify_relationships(
        cls,
        content: str,
        concepts: List[str]
    ) -> List[Tuple[str, str, EdgeType]]:
        """
        Identify relationships between concepts.
        
        Args:
            content: Educational content
            concepts: List of concepts
        
        Returns:
            List of (source, target, edge_type) tuples
        """
        relationships = []
        sentences = re.split(r'[.!?]+', content)
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            concepts_in_sentence = [c for c in concepts if c.lower() in sentence_lower]
            
            if len(concepts_in_sentence) >= 2:
                relationships.extend(
                    cls._extract_relationship_from_sentence(sentence_lower, concepts_in_sentence)
                )
        
        logger.debug(f"Identified {len(relationships)} relationships")
        return relationships


class ConceptMapService:
    """
    Service for generating concept maps from educational content.
    """
    
    def __init__(self):
        """Initialize concept map service."""
        self.extractor = ConceptExtractor()
        logger.info("ConceptMapService initialized")
    
    def generate_concept_map(
        self,
        content: str,
        title: str,
        subject: str,
        grade: int,
        max_nodes: int = 20
    ) -> ConceptMap:
        """
        Generate concept map from content.
        
        Args:
            content: Educational content
            title: Title of the content
            subject: Subject area
            grade: Grade level
            max_nodes: Maximum number of nodes
        
        Returns:
            ConceptMap object
        """
        if not content or len(content.strip()) == 0:
            raise ValueError("Content cannot be empty")
        
        logger.info(f"Generating concept map for '{title}' (grade {grade})")
        
        # Extract concepts
        concepts = self.extractor.extract_concepts(content, subject)
        
        # Limit concepts based on max_nodes
        if len(concepts) > max_nodes:
            # Keep most important concepts (first occurrence, capitalized, etc.)
            concepts = concepts[:max_nodes]
        
        # Identify relationships
        relationships = self.extractor.identify_relationships(content, concepts)
        
        # Build graph
        nodes = self._create_nodes(concepts, title)
        edges = self._create_edges(relationships, nodes)
        
        # Determine root node (usually the title or first concept)
        root_node_id = nodes[0].id if nodes else "root"
        
        # Calculate hierarchy levels
        self._calculate_levels(nodes, edges, root_node_id)
        
        concept_map = ConceptMap(
            title=title,
            subject=subject,
            grade=grade,
            nodes=nodes,
            edges=edges,
            root_node_id=root_node_id,
            metadata={
                'total_concepts': len(nodes),
                'total_relationships': len(edges),
                'max_level': max(n.level for n in nodes) if nodes else 0
            }
        )
        
        logger.info(
            f"Generated concept map: {len(nodes)} nodes, {len(edges)} edges, "
            f"{concept_map.metadata['max_level']} levels"
        )
        
        return concept_map
    
    def _create_nodes(self, concepts: List[str], title: str) -> List[ConceptNode]:
        """Create nodes from concepts."""
        nodes = []
        
        # Root node (main concept)
        nodes.append(ConceptNode(
            id="node_0",
            label=title,
            type=NodeType.MAIN_CONCEPT,
            level=0,
            importance=1.0,
            metadata={}
        ))
        
        # Create nodes for concepts
        for i, concept in enumerate(concepts, start=1):
            node_type = NodeType.SUB_CONCEPT if i <= 5 else NodeType.SUPPORTING_DETAIL
            importance = 1.0 - (i / len(concepts))  # Decreasing importance
            
            nodes.append(ConceptNode(
                id=f"node_{i}",
                label=concept,
                type=node_type,
                level=1,  # Will be recalculated
                importance=importance,
                metadata={}
            ))
        
        return nodes
    
    def _create_edges(
        self,
        relationships: List[Tuple[str, str, EdgeType]],
        nodes: List[ConceptNode]
    ) -> List[ConceptEdge]:
        """Create edges from relationships."""
        edges = []
        
        # Create node label to ID mapping
        label_to_id = {node.label: node.id for node in nodes}
        
        # Create edges from relationships
        for source_label, target_label, edge_type in relationships:
            source_id = label_to_id.get(source_label)
            target_id = label_to_id.get(target_label)
            
            if source_id and target_id:
                edges.append(ConceptEdge(
                    source_id=source_id,
                    target_id=target_id,
                    edge_type=edge_type,
                    label=edge_type.value.replace('_', ' '),
                    weight=0.8
                ))
        
        # Connect to root if no incoming edges
        root_id = nodes[0].id if nodes else None
        if root_id:
            nodes_with_edges = {e.target_id for e in edges}
            for node in nodes[1:]:  # Skip root
                if node.id not in nodes_with_edges:
                    edges.append(ConceptEdge(
                        source_id=root_id,
                        target_id=node.id,
                        edge_type=EdgeType.HAS,
                        label="includes",
                        weight=0.5
                    ))
        
        return edges
    
    def _build_adjacency_list(self, edges: List[ConceptEdge]) -> Dict[str, List[str]]:
        """Build adjacency list from edges."""
        adjacency: Dict[str, List[str]] = defaultdict(list)
        for edge in edges:
            adjacency[edge.source_id].append(edge.target_id)
        return adjacency
    
    def _find_and_set_root_level(self, nodes: List[ConceptNode], root_id: str):
        """Find root node and set its level to 0."""
        for node in nodes:
            if node.id == root_id:
                node.level = 0
                break
    
    def _update_node_level(self, nodes: List[ConceptNode], node_id: str, level: int):
        """Update the level of a specific node."""
        for node in nodes:
            if node.id == node_id:
                node.level = level
                break
    
    def _calculate_levels(
        self,
        nodes: List[ConceptNode],
        edges: List[ConceptEdge],
        root_id: str
    ):
        """Calculate hierarchy levels using BFS."""
        adjacency = self._build_adjacency_list(edges)
        
        # Initialize BFS
        visited: Set[str] = {root_id}
        queue: List[Tuple[str, int]] = [(root_id, 0)]
        self._find_and_set_root_level(nodes, root_id)
        
        # BFS traversal
        while queue:
            current_id, level = queue.pop(0)
            
            for neighbor_id in adjacency[current_id]:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, level + 1))
                    self._update_node_level(nodes, neighbor_id, level + 1)
    
    def export_to_svg(self, concept_map: ConceptMap, output_path: str) -> str:
        """
        Export concept map to SVG.
        
        Args:
            concept_map: ConceptMap to export
            output_path: Path for SVG file
        
        Returns:
            SVG string
        """
        logger.info(f"Exporting concept map to SVG: {output_path}")
        
        # Simple SVG generation
        width = 800
        height = 600
        node_radius = 40
        
        svg_parts = [
            f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
            '<defs>',
            '<style>',
            '.node { fill: #4A90E2; stroke: #333; stroke-width: 2; }',
            '.node-text { fill: white; font-family: Arial; font-size: 12px; text-anchor: middle; }',
            '.edge { stroke: #999; stroke-width: 2; fill: none; marker-end: url(#arrow); }',
            '.edge-label { fill: #666; font-family: Arial; font-size: 10px; }',
            '</style>',
            '<marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">',
            '<path d="M0,0 L0,6 L9,3 z" fill="#999" />',
            '</marker>',
            '</defs>',
        ]
        
        # Calculate node positions (simple tree layout)
        node_positions = self._calculate_node_positions(concept_map, width, height)
        
        # Draw edges
        svg_parts.append('<g id="edges">')
        for edge in concept_map.edges:
            source_pos = node_positions.get(edge.source_id)
            target_pos = node_positions.get(edge.target_id)
            
            if source_pos and target_pos:
                svg_parts.append(
                    f'<line class="edge" x1="{source_pos[0]}" y1="{source_pos[1]}" '
                    f'x2="{target_pos[0]}" y2="{target_pos[1]}" />'
                )
                
                # Edge label
                mid_x = (source_pos[0] + target_pos[0]) / 2
                mid_y = (source_pos[1] + target_pos[1]) / 2
                svg_parts.append(
                    f'<text class="edge-label" x="{mid_x}" y="{mid_y}">{edge.label}</text>'
                )
        svg_parts.append('</g>')
        
        # Draw nodes
        svg_parts.append('<g id="nodes">')
        for node in concept_map.nodes:
            pos = node_positions.get(node.id)
            if pos:
                # Node circle
                svg_parts.append(
                    f'<circle class="node" cx="{pos[0]}" cy="{pos[1]}" r="{node_radius}" />'
                )
                
                # Node label
                svg_parts.append(
                    f'<text class="node-text" x="{pos[0]}" y="{pos[1] + 5}">'
                    f'{node.label[:15]}</text>'
                )
        svg_parts.append('</g>')
        
        svg_parts.append('</svg>')
        
        svg_string = '\n'.join(svg_parts)
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(svg_string)
        
        logger.info(f"SVG exported successfully: {output_path}")
        return svg_string
    
    def _calculate_node_positions(
        self,
        concept_map: ConceptMap,
        width: int,
        height: int
    ) -> Dict[str, Tuple[int, int]]:
        """Calculate node positions for tree layout."""
        positions: Dict[str, Tuple[int, int]] = {}
        
        # Group nodes by level
        levels: Dict[int, List[ConceptNode]] = defaultdict(list)
        for node in concept_map.nodes:
            levels[node.level].append(node)
        
        max_level = max(levels.keys()) if levels else 0
        level_height = height // (max_level + 2)
        
        # Position nodes
        for level, nodes_at_level in levels.items():
            y = level_height * (level + 1)
            node_count = len(nodes_at_level)
            
            if node_count == 1:
                x = width // 2
                positions[nodes_at_level[0].id] = (x, y)
            else:
                spacing = width // (node_count + 1)
                for i, node in enumerate(nodes_at_level):
                    x = spacing * (i + 1)
                    positions[node.id] = (x, y)
        
        return positions
    
    def export_to_json(self, concept_map: ConceptMap, output_path: str) -> str:
        """Export concept map to JSON."""
        data = {
            'title': concept_map.title,
            'subject': concept_map.subject,
            'grade': concept_map.grade,
            'root_node_id': concept_map.root_node_id,
            'metadata': concept_map.metadata,
            'nodes': [
                {
                    'id': n.id,
                    'label': n.label,
                    'type': n.type.value,
                    'level': n.level,
                    'importance': n.importance,
                    'metadata': n.metadata
                }
                for n in concept_map.nodes
            ],
            'edges': [
                {
                    'source': e.source_id,
                    'target': e.target_id,
                    'type': e.edge_type.value,
                    'label': e.label,
                    'weight': e.weight
                }
                for e in concept_map.edges
            ]
        }
        
        json_string = json.dumps(data, indent=2)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json_string)
        
        logger.info(f"JSON exported successfully: {output_path}")
        return json_string


if __name__ == "__main__":
    # Example usage
    sample_content = """
    Photosynthesis is the process by which plants convert light energy into chemical energy.
    Plants have chloroplasts that contain chlorophyll. Chlorophyll absorbs light energy.
    This energy is used to convert carbon dioxide and water into glucose and oxygen.
    The glucose provides energy for the plant. Oxygen is released as a byproduct.
    Photosynthesis requires sunlight, water, and carbon dioxide. It produces glucose and oxygen.
    """
    
    service = ConceptMapService()
    
    logger.info("Concept Map Generator Demo")
    logger.info("=" * 60)
    
    # Generate concept map
    concept_map = service.generate_concept_map(
        content=sample_content,
        title="Photosynthesis",
        subject="Science",
        grade=8,
        max_nodes=15
    )
    
    logger.info(f"\nConcept Map: {concept_map.title}")
    logger.info(f"Subject: {concept_map.subject}, Grade: {concept_map.grade}")
    logger.info(f"Nodes: {len(concept_map.nodes)}, Edges: {len(concept_map.edges)}")
    logger.info(f"Max Level: {concept_map.metadata['max_level']}")
    
    # Export to SVG
    svg_path = "data/cache/concept_map.svg"
    service.export_to_svg(concept_map, svg_path)
    logger.info(f"\nSVG exported to: {svg_path}")
    
    # Export to JSON
    json_path = "data/cache/concept_map.json"
    service.export_to_json(concept_map, json_path)
    logger.info(f"JSON exported to: {json_path}")
