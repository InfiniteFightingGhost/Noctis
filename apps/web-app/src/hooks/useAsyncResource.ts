import { useEffect, useRef, useState } from "react";

type AsyncState<T> = {
  loading: boolean;
  data: T | null;
  error: string | null;
};

export function useAsyncResource<T>(loader: () => Promise<T>) {
  const requestCounter = useRef(0);
  const [state, setState] = useState<AsyncState<T>>({
    loading: true,
    data: null,
    error: null,
  });

  useEffect(() => {
    requestCounter.current += 1;
    const requestId = requestCounter.current;
    let mounted = true;

    setState({ loading: true, data: null, error: null });
    loader()
      .then((data) => {
        if (mounted && requestId === requestCounter.current) {
          setState({ loading: false, data, error: null });
        }
      })
      .catch((error: unknown) => {
        if (mounted && requestId === requestCounter.current) {
          setState({
            loading: false,
            data: null,
            error: error instanceof Error ? error.message : "Unexpected error",
          });
        }
      });

    return () => {
      mounted = false;
    };
  }, [loader]);

  return state;
}
