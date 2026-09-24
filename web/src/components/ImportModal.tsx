import { Alert, Button, List, Modal, Stack, Text } from "@mantine/core";
import { Dropzone } from "@mantine/dropzone";
import { useDisclosure } from "@mantine/hooks";
import type { ImportResult } from "../api/client";
import { useImport } from "../api/hooks";

type Issue = ImportResult["notFound"][number];

export default function ImportModal() {
  const [opened, { open, close }] = useDisclosure(false);
  const imp = useImport();
  const r = imp.data;
  const sections: [string, Issue[]][] = r
    ? [["Не найдены в каталоге", r.notFound], ["Ошибки в строках", r.errors],
       ["Вне CF-ядра — не влияют на рекомендации", r.outsideCore]]
    : [];
  return (
    <>
      <Button size="xs" onClick={() => { imp.reset(); open(); }}>Импорт</Button>
      <Modal opened={opened} onClose={close} title="Импорт оценок" size="lg">
        <Stack>
          <Text size="sm" c="dimmed">
            CSV в нашем формате (goodreads_work_id, rating, status) или экспорт Goodreads (My Books → Export).
            Оценки книг из файла перезапишут текущие, остальные останутся.
          </Text>
          <Dropzone onDrop={(files) => imp.mutate(files[0])} accept={["text/csv", "application/vnd.ms-excel"]}
                    maxFiles={1} loading={imp.isPending}>
            <Text ta="center" py="xl">Перетащите CSV сюда или нажмите, чтобы выбрать</Text>
          </Dropzone>
          {imp.error && <Alert color="red">{imp.error.message}</Alert>}
          {r && (
            <Alert color="green" title={`Добавлено ${r.added}, обновлено ${r.updated}, пропущено без оценки ${r.skipped}`}>
              {sections.filter(([, items]) => items.length > 0).map(([title, items]) => (
                <div key={title}>
                  <Text fw={600} size="sm" mt="xs">{title}</Text>
                  <List size="sm">
                    {items.map((i) => <List.Item key={i.line}>стр. {i.line}: {i.book} — {i.reason}</List.Item>)}
                  </List>
                </div>
              ))}
            </Alert>
          )}
        </Stack>
      </Modal>
    </>
  );
}
