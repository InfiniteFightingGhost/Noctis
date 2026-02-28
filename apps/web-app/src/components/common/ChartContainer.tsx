import type { ReactNode } from "react";

type ChartContainerProps = {
  title: string;
  subtitle?: string;
  children: ReactNode;
};

export function ChartContainer({ title, subtitle, children }: ChartContainerProps) {
  return (
    <section className="chart-card">
      <div className="chart-heading">
        <h2>{title}</h2>
        {subtitle ? <p>{subtitle}</p> : null}
      </div>
      {children}
    </section>
  );
}
