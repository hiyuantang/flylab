/** Coalesce display refreshes without cancelling work or queuing old frames. */
export function activityRefresh(
  load: (stamp: string) => Promise<void>,
  visible: () => boolean,
  retryMs = 1000,
) {
  let latest: string | null = null;
  let completed: string | null = null;
  let busy = false;
  let disposed = false;
  let retry: ReturnType<typeof setTimeout> | undefined;
  const drain = async () => {
    if (
      disposed ||
      busy ||
      retry ||
      !visible() ||
      latest === null ||
      latest === completed
    )
      return;
    const requested = latest;
    busy = true;
    try {
      await load(requested);
      completed = requested;
    } catch {
      if (!disposed)
        retry = setTimeout(() => {
          retry = undefined;
          void drain();
        }, retryMs);
    } finally {
      busy = false;
      if (!disposed && !retry && latest !== completed) void drain();
    }
  };
  return {
    request(stamp: string) {
      latest = stamp;
      void drain();
    },
    dispose() {
      disposed = true;
      clearTimeout(retry);
    },
  };
}
