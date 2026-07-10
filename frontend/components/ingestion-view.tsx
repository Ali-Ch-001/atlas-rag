"use client";

import { useEffect, useState } from "react";
import { AlertTriangle, ArrowRight, Check, Clock3, FileSearch, Layers3, RefreshCw, ScanText, Split, UploadCloud } from "lucide-react";
import type { IngestionSummary } from "@/lib/types";
import { getOperationsData } from "@/lib/api";
import { EmptyState, MetricCard, PageHeader, SectionPanel, StatusBadge } from "@/components/ui";

function formatRate(total: number): string {
  if (total === 0) return "0/h";
  return `${total.toLocaleString()}/h`;
}

export function IngestionView() {
  const [data, setData] = useState<IngestionSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [notice, setNotice] = useState("");

  async function load() {
    setLoading(true);
    try {
      const result = await getOperationsData();
      setData(result.ingestion);
    } catch {
      setData(null);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { void load(); }, []);

  const totalQueue = data
    ? Object.values(data.queue_depth).reduce((sum, count) => sum + count, 0)
    : 0;
  const totalFailed = data
    ? Object.values(data.retry_queue).reduce((sum, count) => sum + count, 0)
    : 0;
  const todayRate = data ? data.today_processed : 0;

  return (
    <div>
      <PageHeader
        eyebrow="Pipeline / Ingestion"
        title="Document processing"
        description="Trace document flow from receipt through durable vector indexing and recover failed work."
        actions={<button className="btn-secondary" onClick={() => void load()} disabled={loading}><RefreshCw size={16} className={loading ? "animate-spin" : ""} aria-hidden="true" /> Refresh</button>}
      />
      <div className="mx-auto max-w-[1500px] space-y-6 p-4 sm:p-6 lg:p-8">
        {notice && <div role="status" className="flex items-center gap-2 border border-accent/40 bg-accent/10 px-4 py-3 text-sm text-[#9af0ba]"><Check size={16} aria-hidden="true" />{notice}</div>}

        {!data && !loading && (
          <div className="panel-raised p-6">
            <EmptyState icon={Layers3} title="Ingestion telemetry unavailable" description="The pipeline summary endpoint is not connected. Start the ingestion worker and refresh." />
          </div>
        )}

        <section className="grid grid-cols-1 gap-px border border-line bg-line sm:grid-cols-2 xl:grid-cols-4" aria-label="Ingestion queue metrics">
          <MetricCard label="Queue depth" value={String(totalQueue)} detail={data ? "Live pipeline occupancy" : "Pipeline worker offline"} icon={Layers3} status={data ? "positive" : undefined} />
          <MetricCard label="Processed today" value={String(todayRate)} detail={formatRate(todayRate)} icon={RefreshCw} status="positive" />
          <MetricCard label="Active documents" value={String(data?.active_documents ?? "—")} detail={data ? `Across ${Object.keys(data.document_states).length} states` : "Not connected"} icon={Clock3} />
          <MetricCard label="Failed / 24h" value={String(data?.failed_24h ?? "—")} detail={data ? `${((data.failed_24h / Math.max(data.today_processed + data.failed_24h, 1)) * 100).toFixed(1)}% of received` : "Not connected"} icon={AlertTriangle} status={totalFailed > 0 ? "warning" : "positive"} />
        </section>

        {data && (
          <SectionPanel title="Processing pipeline" eyebrow="Real-time stage occupancy" action={<StatusBadge state={totalQueue > 0 ? "operational" : "unknown"} label={totalQueue > 0 ? "Live data" : "Empty queue"} />}>
            <ol className="grid gap-px bg-line md:grid-cols-5">
              {(data.pipeline_stages.length > 0 ? data.pipeline_stages : [
                { name: "Ingestion", count: totalQueue, completed: data.today_processed, failed: totalFailed },
              ]).slice(0, 5).map((stage, index) => {
                const names: Record<string, string> = {
                  full_pipeline: "Processing", quarantine: "Received", scan: "Scanned",
                  parse: "Extracted", ocr: "OCR", chunk: "Chunked", embed: "Embedded", index: "Indexed",
                };
                return (
                  <li key={stage.name} className="relative bg-surface-1 p-4 sm:p-5">
                    <div className="flex items-center justify-between">
                      {index === 0 ? <UploadCloud size={19} className="text-accent" aria-hidden="true" />
                        : index === 1 ? <ScanText size={19} className="text-accent" aria-hidden="true" />
                        : index === 2 ? <Split size={19} className="text-accent" aria-hidden="true" />
                        : index === 3 ? <Layers3 size={19} className="text-accent" aria-hidden="true" />
                        : <FileSearch size={19} className="text-accent" aria-hidden="true" />}
                      <span className="data-value text-xl font-semibold">{stage.count}</span>
                    </div>
                    <p className="mt-5 font-mono text-sm font-semibold">{String(index + 1).padStart(2, "0")} / {names[stage.name] ?? stage.name}</p>
                    <p className="mt-1 text-xs text-ink-faint">{stage.completed} completed{stage.failed > 0 ? ` / ${stage.failed} failed` : ""}</p>
                    {index < Math.min((data.pipeline_stages.length || 1), 5) - 1 && (
                      <ArrowRight className="absolute -right-3 top-1/2 z-10 hidden -translate-y-1/2 bg-surface-1 text-line-strong md:block" size={22} aria-hidden="true" />
                    )}
                  </li>
                );
              })}
            </ol>
          </SectionPanel>
        )}

        {data && data.recent_failures.length > 0 && (
          <SectionPanel title="Failed documents" eyebrow="Requires intervention" action={<span className="font-mono text-xs text-danger">{data.recent_failures.length} unresolved</span>}>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[760px] text-left text-sm">
                <thead className="border-b border-line bg-surface-2 font-mono text-[10px] uppercase tracking-[0.12em] text-ink-faint"><tr><th className="px-5 py-3 font-medium">Job</th><th className="px-4 py-3 font-medium">Stage</th><th className="px-4 py-3 font-medium">Failure</th><th className="px-5 py-3 text-right font-medium">Time</th></tr></thead>
                <tbody className="divide-y divide-line">
                  {data.recent_failures.map((failure) => (
                    <tr key={failure.job_id} className="transition-colors hover:bg-surface-2">
                      <td className="px-5 py-4 font-mono text-xs text-ink-muted">{failure.job_id}</td>
                      <td className="px-4 py-4"><StatusBadge state="down" label={failure.stage} /></td>
                      <td className="max-w-sm px-4 py-4 text-xs text-ink-muted">{failure.reason}</td>
                      <td className="px-5 py-4 text-right font-mono text-xs text-ink-faint">{failure.at ? new Date(failure.at).toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", timeZone: "UTC" }) + " UTC" : "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </SectionPanel>
        )}
      </div>
    </div>
  );
}
