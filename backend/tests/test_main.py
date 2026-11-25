from app.main import PROJECT_NAME, app
from backend.tests.http_client import InProcessClient

client = InProcessClient(app)


def test_root_returns_welcome_and_project_name() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to the project API.",
        "project": PROJECT_NAME,
    }


def test_health_reports_api_is_running() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
