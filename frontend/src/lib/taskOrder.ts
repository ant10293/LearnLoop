export interface PartialOrder {
  ranked: string[][];
  unordered?: string[];
  ambiguous_ids?: string[];
}

export function clipPosition(order: PartialOrder, id: string) {
  const group = order.ranked.findIndex((items) => items.includes(id));
  return {
    placed: group >= 0,
    tied: group >= 0 && order.ranked[group].length > 1,
    ambiguous: (order.ambiguous_ids ?? []).includes(id) || (order.unordered ?? []).includes(id),
  };
}
