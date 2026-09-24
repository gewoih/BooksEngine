import { notifications } from "@mantine/notifications";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, unwrap, type Book, type ImportResult, type MyRating } from "./client";

export const keys = {
  session: ["session"] as const,
  users: ["users"] as const,
  library: (q: string, page: number) => ["library", q, page] as const,
  myRatings: ["my-ratings"] as const,
  recommendations: (n: number) => ["recommendations", n] as const,
  work: (id: number) => ["work", id] as const,
};

export function useSession() {
  return useQuery({
    queryKey: keys.session,
    queryFn: async () => {
      const r = await api.GET("/api/session");
      return r.response.status === 401 ? null : unwrap(r);
    },
  });
}

export function useUsers() {
  return useQuery({ queryKey: keys.users, queryFn: async () => unwrap(await api.GET("/api/users")) });
}

export function useLogin() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (userId: number) => unwrap(await api.POST("/api/session", { body: { userId } })),
    onSuccess: () => qc.invalidateQueries(),               // сменился пользователь — всё чужое
  });
}

export function useCreateUser() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (name: string) => unwrap(await api.POST("/api/users", { body: { name } })),
    onSuccess: () => qc.invalidateQueries({ queryKey: keys.users }),
  });
}

export function useLogout() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async () => { await api.DELETE("/api/session"); },
    onSuccess: () => qc.invalidateQueries(),
  });
}

export function useLibrary(q: string, page: number) {
  return useQuery({
    queryKey: keys.library(q, page),
    queryFn: async () => unwrap(await api.GET("/api/library", { params: { query: { q: q || undefined, page } } })),
    placeholderData: (prev) => prev,                       // список не мигает при наборе запроса
  });
}

export function useMyRatings() {
  return useQuery({ queryKey: keys.myRatings, queryFn: async () => unwrap(await api.GET("/api/me/ratings")) });
}

type RateArgs = { workId: number; value?: number; dnf?: boolean; remove?: boolean; book: Book };

/** Оценка сразу видна в панели и списке (оптимистично); при ошибке — откат и уведомление. */
export function useRate() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ workId, value, dnf, remove }: RateArgs) => {
      if (remove) {
        await api.DELETE("/api/me/ratings/{workId}", { params: { path: { workId } } });
        return null;
      }
      return unwrap(await api.PUT("/api/me/ratings/{workId}", {
        params: { path: { workId } }, body: { value: value ?? null, dnf: !!dnf },
      }));
    },
    onMutate: async ({ workId, value, dnf, remove, book }) => {
      await qc.cancelQueries({ queryKey: keys.myRatings });
      const prev = qc.getQueryData<MyRating[]>(keys.myRatings);
      const updated: Book = { ...book, myRating: remove ? null : dnf ? 1 : value ?? null, myDnf: !!dnf && !remove };
      qc.setQueryData<MyRating[]>(keys.myRatings, (old) => {
        const rest = (old ?? []).filter((r) => r.book.workId !== workId);
        return remove ? rest : [{ book: updated, updatedAt: new Date().toISOString() }, ...rest];
      });
      qc.setQueriesData<{ items: Book[] }>({ queryKey: ["library"] }, (old) =>
        old && { ...old, items: old.items.map((b) => (b.workId === workId ? updated : b)) });
      return { prev };
    },
    onError: (e, _v, ctx) => {
      qc.setQueryData(keys.myRatings, ctx?.prev);
      notifications.show({ color: "red", title: "Оценка не сохранилась", message: e.message });
    },
    onSettled: () => {
      qc.invalidateQueries({ queryKey: keys.myRatings });
      qc.invalidateQueries({ queryKey: ["library"] });
      qc.invalidateQueries({ queryKey: ["recommendations"] });
      qc.invalidateQueries({ queryKey: ["work"] });
    },
  });
}

export function useRecommendations(n: number) {
  return useQuery({
    queryKey: keys.recommendations(n),
    queryFn: async () => unwrap(await api.GET("/api/me/recommendations", { params: { query: { n } } })),
  });
}

export function useWork(id: number | null) {
  return useQuery({
    queryKey: keys.work(id ?? 0),
    enabled: id != null,
    queryFn: async () => unwrap(await api.GET("/api/works/{id}", { params: { path: { id: id! } } })),
  });
}

/** multipart с openapi-fetch неудобен — обычный fetch; тип ответа — из схемы. */
export function useImport() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (file: File) => {
      const body = new FormData();
      body.append("file", file);
      const r = await fetch("/api/me/import", { method: "POST", body });
      if (!r.ok) {
        const text = await r.text();
        throw new Error((text.startsWith('"') ? JSON.parse(text) : text) || `Ошибка ${r.status}`);
      }
      return (await r.json()) as ImportResult;
    },
    onSuccess: () => qc.invalidateQueries(),
  });
}
