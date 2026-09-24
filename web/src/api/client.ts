import createClient from "openapi-fetch";
import type { components, paths } from "./schema";

export const api = createClient<paths>({ baseUrl: "" });
export type Book = components["schemas"]["BookDto"];
export type User = components["schemas"]["UserDto"];
export type MyRating = components["schemas"]["MyRatingDto"];
export type ImportResult = components["schemas"]["ImportResultDto"];

/** Ответ API → данные или исключение с текстом ошибки сервера (он на русском). */
export function unwrap<T>(r: { data?: T; error?: unknown; response: Response }): T {
  if (r.error !== undefined || r.data === undefined) {
    const text = typeof r.error === "string" ? r.error : (r.error as { detail?: string } | undefined)?.detail;
    throw new Error(text ?? `Ошибка ${r.response.status}`);
  }
  return r.data;
}
