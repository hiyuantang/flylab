/** Invalidate reads started before an action or a controller change. */
export function trainingRequestGate() {
  let generation = 0;
  return {
    begin: () => ++generation,
    invalidate: () => {
      generation += 1;
    },
    accepts: (token: number) => token === generation,
  };
}
