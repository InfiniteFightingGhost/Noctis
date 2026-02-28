const labels = ["Wake", "Light", "Deep", "REM"];

type TransitionMatrixProps = {
  matrix: number[][];
};

export function TransitionMatrix({ matrix }: TransitionMatrixProps) {
  if (matrix.length === 0 || matrix.every((row) => row.length === 0)) {
    return (
      <div className="chart-empty" role="status" aria-live="polite">
        No data for now.
      </div>
    );
  }

  const flattened = matrix.flat();
  const max = Math.max(...flattened, 1);

  return (
    <div className="transition-matrix" role="table" aria-label="transition matrix">
      <div className="matrix-row header" role="row">
        <span aria-hidden="true" />
        {labels.map((label) => (
          <span key={label}>{label}</span>
        ))}
      </div>
      {matrix.map((row, rowIndex) => (
        <div className="matrix-row" key={`row-${labels[rowIndex] ?? rowIndex}`} role="row">
          <span>{labels[rowIndex] ?? `S${rowIndex + 1}`}</span>
          {row.map((cell, cellIndex) => (
            <span
              key={`cell-${rowIndex}-${cellIndex}`}
              style={{
                backgroundColor: `color-mix(in srgb, var(--accent) ${14 + (cell / max) * 72}%, transparent)`,
              }}
              className="matrix-cell"
            >
              {cell}
            </span>
          ))}
        </div>
      ))}
      <p className="chart-footnote">Matrix intensity represents transition frequency (darker = more transitions).</p>
    </div>
  );
}
