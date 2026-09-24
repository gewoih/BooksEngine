import { Center, Divider, Loader, Pagination, Stack, Text, TextInput } from "@mantine/core";
import { useDebouncedValue } from "@mantine/hooks";
import { useState } from "react";
import { useLibrary } from "../api/hooks";
import BookRow from "../components/BookRow";
import MyRatingsPanel from "../components/MyRatingsPanel";

export default function LibraryPage() {
  const [q, setQ] = useState("");
  const [page, setPage] = useState(1);
  const [query] = useDebouncedValue(q.trim(), 300);
  const lib = useLibrary(query, page);
  const pages = lib.data ? Math.max(1, Math.ceil(lib.data.total / lib.data.pageSize)) : 1;
  return (
    <>
      <MyRatingsPanel />
      <TextInput placeholder="Название или автор" value={q} mb="sm"
                 onChange={(e) => { setQ(e.currentTarget.value); setPage(1); }} />
      {lib.isLoading ? <Center><Loader /></Center> : (
        <>
          <Text size="sm" c="dimmed" mb="xs">Найдено: {lib.data?.total ?? 0}</Text>
          <Stack gap={0}>
            {lib.data?.items.map((b) => (<div key={b.workId}><BookRow book={b} /><Divider /></div>))}
          </Stack>
          <Center mt="md"><Pagination total={pages} value={page} onChange={setPage} siblings={1} /></Center>
        </>
      )}
    </>
  );
}
