import { AppShell, Container } from "@mantine/core";
import { Route, Routes } from "react-router";
import AppHeader from "./components/AppHeader";
import UserGate from "./components/UserGate";
import WorkDrawer from "./components/WorkDrawer";
import RecommendationsPage from "./pages/RecommendationsPage";
import LibraryPage from "./pages/LibraryPage";

export default function App() {
  return (
    <UserGate>
      <AppShell header={{ height: 60 }} padding="md">
        <AppShell.Header>
          <AppHeader />
        </AppShell.Header>
        <AppShell.Main>
          <Container size="lg">
            <Routes>
              <Route path="/" element={<LibraryPage />} />
              <Route path="/recommendations" element={<RecommendationsPage />} />
            </Routes>
            <WorkDrawer />
          </Container>
        </AppShell.Main>
      </AppShell>
    </UserGate>
  );
}
