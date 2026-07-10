"use client";

import { useEffect, useState } from "react";
import type { Metadata } from "next";
import { AlertTriangle, Boxes, Clock3, DatabaseZap, Gauge, HardDrive, Network, RefreshCw, Server, TimerReset } from "lucide-react";
import type { OperationsData } from "@/lib/types";
import { getOperationsData } from "@/lib/api";
import { EmptyState, MetricCard, PageHeader, SectionPanel, StatusBadge } from "@/components/ui";

export default function OperationsPage() {
  const [data, setData] = useState<OperationsData | null>(null);
  const [loading, setLoading] = useState(true);

  async function load() {
    setLoading(true);
    try {
      setData(await getOperationsData());
    } catch {
      setData(null);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { void load(); }, []);

  const p95 = data?.sloBudgets?.[0]?.current ?? "N/A";
  const cacheRate = data?.sloBudgets?.[1]?.current ?? "N/A";
  const p50 = data?.sloBudgets?.[2]?.current ?? "N/A";
  const totalSloUsed = data?.sloBudgets?.reduce((sum, slo) => sum + slo.used, 0) ?? 0;

  return (
    <div>
      <PageHeader
        eyebrow="Reliability / Operations"
        title="Infrastructure & SLOs"
        description="Inspect service objectives, queue pressure, dependency health, and active operational alerts."
        actions={<button className="btn-secondary" onClick={() => void load()} disabled={loading}><RefreshCw size={16} className={loading ? "animate-spin" : ""} aria-hidden="true" /> Refresh</button>}
      />
      <div className="mx-auto max-w-[1500px] space-y-6 p-4 sm:p-6 lg:p-8">
        {data?.demo && (
          <div className="panel-raised p-6">
            <EmptyState icon={HardDrive} title="Telemetry endpoint not connected" description="Start the API with live dependencies and metrics to populate this page with real data." />
          </div>
        )}

        <section className="grid grid-cols-1 gap-px border border-line bg-line sm:grid-cols-2 xl:grid-cols-4" aria-label="SLO metrics">
          <MetricCard label="Response P95" value={p95} detail={p95 === "N/A" ? "No live data" : "Live retrieval telemetry"} icon={Gauge} status={p95 !== "N/A" ? "positive" : "unknown"} />
          <MetricCard label="Cache hit rate" value={cacheRate} detail={cacheRate === "N/A" ? "No live data" : "Rolling 24-hour window"} icon={DatabaseZap} status={cacheRate !== "N/A" ? "positive" : "unknown"} />
          <MetricCard label="Retrieval P50" value={p50} detail={p50 === "N/A" ? "No live data" : "Live retrieval telemetry"} icon={Server} status={p50 !== "N/A" ? "positive" : "unknown"} />
          <MetricCard label="SLO headroom" value={`${Math.max(0, 100 - totalSloUsed)}%`} detail="Combined budget remaining" icon={Clock3} status={totalSloUsed < 70 ? "positive" : "warning"} />
        </section>

        <div className="grid gap-6 xl:grid-cols-[1fr_1.15fr]">
          <SectionPanel title="SLO budget" eyebrow="Rolling 30 days" action={<StatusBadge state={totalSloUsed < 70 ? "operational" : "degraded"} label={totalSloUsed < 70 ? "Within budget" : "Budget at risk"} />}>
            <div className="space-y-6 p-5">
              {(data?.sloBudgets ?? []).map((slo) => (
                <div key={slo.label}>
                  <div className="flex items-end justify-between gap-4">
                    <div><p className="text-sm font-medium">{slo.label}</p><p className="mt-1 text-xs text-ink-faint">Objective {slo.target}</p></div>
                    <p className="data-value text-sm">{slo.current}</p>
                  </div>
                  <div className="mt-3 h-2 bg-surface-3" role="progressbar" aria-label={`${slo.label} budget used`} aria-valuenow={slo.used} aria-valuemin={0} aria-valuemax={100}>
                    <div className={`h-full ${slo.used > 80 ? "bg-danger" : "bg-accent"}`} style={{ width: `${slo.used}%` }} />
                  </div>
                  <p className="mt-1.5 text-right font-mono text-[10px] text-ink-faint">{slo.used}% budget consumed</p>
                </div>
              ))}
              {data?.demo && <p className="text-xs text-ink-faint text-center">Budget values are placeholders until the API is connected.</p>}
            </div>
          </SectionPanel>

          <SectionPanel title="Queue pressure" eyebrow="Ingestion pipeline state">
            <div className="grid gap-px bg-line sm:grid-cols-2">
              {[
                { label: "Queue depth", value: data?.ingestion ? Object.values(data.ingestion.queue_depth).reduce((sum: number, count: number) => sum + count, 0) : "—", detail: "Processing pipeline", icon: Boxes },
                { label: "Active documents", value: data?.ingestion?.active_documents ?? "—", detail: "Indexed and searchable", icon: Network },
                { label: "Failed today", value: data?.ingestion?.failed_24h ?? "—", detail: "Needing retry or review", icon: TimerReset },
                { label: "Processed today", value: data?.ingestion?.today_processed ?? "—", detail: "Documents activated", icon: DatabaseZap },
              ].map((queue) => {
                const Icon = queue.icon;
                return (
                  <div className="bg-surface-1 p-5" key={queue.label}>
                    <div className="flex items-start justify-between">
                      <Icon size={18} className="text-ink-faint" aria-hidden="true" />
                      <span className="data-value text-2xl">{String(queue.value)}</span>
                    </div>
                    <p className="mt-5 text-sm font-medium">{queue.label}</p>
                    <p className="mt-1 font-mono text-xs text-ink-faint">{queue.detail}</p>
                  </div>
                );
              })}
            </div>
          </SectionPanel>
        </div>

        <SectionPanel title="Dependencies" eyebrow="External and stateful services">
          <ul className="divide-y divide-line">
            {(data?.dependencies ?? []).map((dependency) => (
              <li key={dependency.name} className="flex flex-wrap items-center gap-3 px-5 py-4">
                <HardDrive size={17} className="text-ink-faint" aria-hidden="true" />
                <div className="min-w-[180px] flex-1">
                  <p className="text-sm font-medium">{dependency.name}</p>
                  <p className="mt-0.5 text-xs text-ink-faint">{dependency.role}</p>
                </div>
                <span className="w-16 font-mono text-xs text-ink-muted">{typeof dependency.latency === "string" ? dependency.latency : `${dependency.latency}ms`}</span>
                <StatusBadge state={dependency.state} />
              </li>
            ))}
          </ul>
          {data?.demo && <p className="p-4 text-xs text-ink-faint text-center border-t border-line">Dependency status requires a connected health endpoint.</p>}
        </SectionPanel>
      </div>
    </div>
  );
}
