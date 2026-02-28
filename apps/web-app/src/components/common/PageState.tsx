type PageStateProps = {
  mode: "loading" | "error";
  message?: string;
};

export function PageState({ mode, message }: PageStateProps) {
  if (mode === "loading") {
    return (
      <div className="page-state loading" role="status" aria-live="polite" aria-label="Loading data">
        <span className="skeleton-line" />
        <span className="skeleton-line" />
        <span className="skeleton-line short" />
      </div>
    );
  }

  return <div className="page-state error">{message ?? "Unable to load data."}</div>;
}
