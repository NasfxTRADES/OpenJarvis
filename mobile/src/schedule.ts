export function humanizeSchedule(
  type: string,
  value: string,
): string {
  if (type === 'interval') {
    const seconds = Number(value);
    if (!Number.isFinite(seconds)) return 'Repeating';
    if (seconds % 3600 === 0) {
      const hours = seconds / 3600;
      return hours === 1 ? 'Every hour' : `Every ${hours} hours`;
    }
    const minutes = Math.round(seconds / 60);
    return minutes === 1 ? 'Every minute' : `Every ${minutes} minutes`;
  }
  if (type === 'once') {
    const when = new Date(value);
    if (Number.isNaN(when.getTime())) return 'Once';
    return `Once on ${when.toLocaleString(undefined, {
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
    })}`;
  }
  if (type === 'cron') {
    const parts = value.trim().split(/\s+/);
    if (parts.length === 5) {
      const [minute, hour, , , dow] = parts;
      const hh = Number(hour);
      const mm = Number(minute);
      const valid = Number.isInteger(hh) && Number.isInteger(mm);
      const time = valid
        ? new Date(2000, 0, 1, hh, mm).toLocaleTimeString(undefined, {
            hour: 'numeric',
            minute: '2-digit',
          })
        : `${hour}:${minute}`;
      if (dow === '*' && valid) return `Every day at ${time}`;
      if ((dow === '1-5' || dow === 'mon-fri') && valid) return `Weekdays at ${time}`;
      if (dow === '0' || dow === '7') return `Sundays at ${time}`;
      if (dow === '1') return `Mondays at ${time}`;
      if (dow === '6') return `Saturdays at ${time}`;
    }
    return `Cron ${value}`;
  }
  return 'Custom schedule';
}

export function relativeTime(iso: string | null): string {
  if (!iso) return '—';
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return '—';
  const diffMs = date.getTime() - Date.now();
  const abs = Math.abs(diffMs);
  const minutes = Math.round(abs / 60000);
  const suffix = diffMs >= 0 ? 'from now' : 'ago';
  if (minutes < 1) return diffMs >= 0 ? 'in a moment' : 'just now';
  if (minutes < 60) return `${minutes} min ${suffix}`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours} h ${suffix}`;
  const days = Math.round(hours / 24);
  if (days < 14) return `${days} d ${suffix}`;
  return date.toLocaleDateString();
}
