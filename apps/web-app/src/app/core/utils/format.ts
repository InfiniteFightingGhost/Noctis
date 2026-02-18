const safeDate = (value?: string | null): Date | null => {
  if (!value) {
    return null;
  }

  const parsed = new Date(value);
  if (Number.isNaN(parsed.valueOf())) {
    return null;
  }

  return parsed;
};

export const formatShortDate = (value?: string | null): string => {
  const parsed = safeDate(value);
  if (!parsed) {
    return "--";
  }

  return new Intl.DateTimeFormat("en-US", {
    weekday: "short",
    month: "short",
    day: "numeric",
  }).format(parsed);
};

export const formatShortTime = (value?: string | null): string => {
  const parsed = safeDate(value);
  if (!parsed) {
    return "--";
  }

  return new Intl.DateTimeFormat("en-US", {
    hour: "2-digit",
    minute: "2-digit",
  }).format(parsed);
};

export const formatMinutesAsClock = (totalMin?: number | null): string => {
  if (totalMin === null || totalMin === undefined || Number.isNaN(totalMin)) {
    return "--";
  }

  const safe = Math.max(0, Math.round(totalMin));
  const hours = Math.floor(safe / 60);
  const minutes = safe % 60;

  return `${hours}:${minutes.toString().padStart(2, "0")}`;
};

export const formatPct = (value?: number | null): string => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "--";
  }

  return `${Math.round(value)}%`;
};
