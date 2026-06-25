import { useEffect, useRef, useState } from 'react'
import * as d3 from 'd3'
import { motion } from 'framer-motion'
import { ClusterSummary } from '../../lib/api'
import { imageUrl } from '../../lib/api'

interface ClusterNode {
  id: string
  cluster_id: number
  x?: number
  y?: number
  fx?: number | null
  fy?: number | null
}

interface ClusterMapProps {
  clusters: ClusterSummary[]
  onSelectCluster: (clusterId: number) => void
  selectedClusterId: number | null
}

const CLUSTER_COLORS = [
  '#C8A882', '#8BA5C0', '#9B8BC0', '#7CB88F', '#C07878',
]

export default function ClusterMap({ clusters, onSelectCluster, selectedClusterId }: ClusterMapProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const [dimensions, setDimensions] = useState({ width: 600, height: 400 })

  useEffect(() => {
    const ro = new ResizeObserver(entries => {
      for (const entry of entries) {
        setDimensions({
          width: entry.contentRect.width,
          height: Math.max(300, entry.contentRect.height),
        })
      }
    })
    if (svgRef.current?.parentElement) ro.observe(svgRef.current.parentElement)
    return () => ro.disconnect()
  }, [])

  useEffect(() => {
    if (!svgRef.current || clusters.length === 0) return
    const { width, height } = dimensions

    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    // Build nodes: each representative image = one node
    const nodes: ClusterNode[] = clusters.flatMap(c =>
      c.representative_paths.slice(0, 5).map((path, i) => ({
        id: `${c.cluster_id}-${i}`,
        cluster_id: c.cluster_id,
        path,
      }))
    )

    // Build cluster group centroids (soft-attract within cluster)
    const clusterCentroids: Record<number, { x: number; y: number }> = {}
    const angleStep = (2 * Math.PI) / Math.max(clusters.length, 1)
    const radius = Math.min(width, height) * 0.3

    clusters.forEach((c, i) => {
      const angle = i * angleStep - Math.PI / 2
      clusterCentroids[c.cluster_id] = {
        x: width / 2 + radius * Math.cos(angle),
        y: height / 2 + radius * Math.sin(angle),
      }
    })

    // Image size
    const imgSize = Math.min(52, width / (clusters.length * 3))

    // Define clip paths for circular images
    const defs = svg.append('defs')
    nodes.forEach(n => {
      defs.append('clipPath')
        .attr('id', `clip-${n.id}`)
        .append('circle')
        .attr('r', imgSize / 2)
    })

    const g = svg.append('g')

    // Cluster halos
    clusters.forEach((c, i) => {
      const center = clusterCentroids[c.cluster_id]
      const color = CLUSTER_COLORS[i % CLUSTER_COLORS.length]
      g.append('circle')
        .attr('cx', center.x)
        .attr('cy', center.y)
        .attr('r', imgSize * 2.5)
        .attr('fill', color)
        .attr('fill-opacity', 0.06)
        .attr('stroke', color)
        .attr('stroke-opacity', 0.2)
        .attr('stroke-width', 1)
    })

    // Cluster labels
    clusters.forEach((c, i) => {
      const center = clusterCentroids[c.cluster_id]
      const color = CLUSTER_COLORS[i % CLUSTER_COLORS.length]
      g.append('text')
        .attr('x', center.x)
        .attr('y', center.y - imgSize * 2.8)
        .attr('text-anchor', 'middle')
        .attr('fill', color)
        .attr('font-size', '11px')
        .attr('font-weight', '600')
        .attr('letter-spacing', '0.06em')
        .attr('font-family', 'Inter, sans-serif')
        .text(c.aesthetic_name.toUpperCase())
    })

    // Force simulation
    const sim = d3.forceSimulation(nodes as any)
      .force('cluster', () => {
        nodes.forEach((n: any) => {
          const center = clusterCentroids[n.cluster_id]
          const dx = center.x - n.x
          const dy = center.y - n.y
          const dist = Math.sqrt(dx * dx + dy * dy) || 1
          const strength = 0.15
          n.vx += (dx / dist) * strength
          n.vy += (dy / dist) * strength
        })
      })
      .force('collision', d3.forceCollide(imgSize * 0.6))
      .force('charge', d3.forceManyBody().strength(-20))
      .alpha(0.8)
      .alphaDecay(0.025)

    // Node groups
    const nodeG = g.selectAll('.node')
      .data(nodes)
      .join('g')
      .attr('class', 'node')
      .style('cursor', 'pointer')
      .on('click', (_, d) => onSelectCluster(d.cluster_id))

    nodeG.append('circle')
      .attr('r', imgSize / 2 + 2)
      .attr('fill', (d, i) => {
        const idx = clusters.findIndex(c => c.cluster_id === d.cluster_id)
        return CLUSTER_COLORS[idx % CLUSTER_COLORS.length]
      })
      .attr('fill-opacity', (d) => selectedClusterId === null || d.cluster_id === selectedClusterId ? 0.3 : 0.1)

    nodeG.append('image')
      .attr('href', (d: any) => imageUrl(d.path))
      .attr('x', -imgSize / 2)
      .attr('y', -imgSize / 2)
      .attr('width', imgSize)
      .attr('height', imgSize)
      .attr('clip-path', d => `url(#clip-${d.id})`)
      .attr('opacity', (d) => selectedClusterId === null || d.cluster_id === selectedClusterId ? 1 : 0.35)
      .attr('preserveAspectRatio', 'xMidYMid slice')

    sim.on('tick', () => {
      nodeG.attr('transform', (d: any) => `translate(${d.x}, ${d.y})`)
    })

    return () => { sim.stop() }
  }, [clusters, dimensions, selectedClusterId, onSelectCluster])

  return (
    <motion.div
      className="cluster-map-wrapper"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.6 }}
    >
      <svg
        ref={svgRef}
        width={dimensions.width}
        height={dimensions.height}
      />
      <style>{`
        .cluster-map-wrapper {
          width: 100%;
          height: 420px;
          background: var(--color-surface);
          border: 1px solid var(--color-border);
          border-radius: var(--radius-lg);
          overflow: hidden;
        }
        .cluster-map-wrapper svg { display: block; }
        .node { transition: opacity 0.2s; }
      `}</style>
    </motion.div>
  )
}
