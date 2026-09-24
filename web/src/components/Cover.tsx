import { Center, Image, Paper, Text } from "@mantine/core";

const SIZE = { m: { w: 50, h: 75 }, l: { w: 160, h: 240 } };

/** Обложка Goodreads; у 12% книг ядра её нет — заглушка с названием. */
export default function Cover({ url, size = "m", title }: { url?: string | null; size?: "m" | "l"; title: string }) {
  const { w, h } = SIZE[size];
  if (!url) {
    return (
      <Paper withBorder w={w} h={h} style={{ flexShrink: 0 }}>
        <Center h="100%" p={4}><Text size="xs" c="dimmed" lineClamp={4} ta="center">{title}</Text></Center>
      </Paper>
    );
  }
  return <Image src={url} w={w} h={h} fit="cover" radius="sm" alt={title} style={{ flexShrink: 0 }} />;
}
