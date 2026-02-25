# Oskín AgTech Backend

Production-ready FastAPI backend for Oskín — an AgTech platform for Kazakhstan farmers.

## Quick Start

```bash
cp .env.example .env
docker-compose up --build
```

API will be available at: http://localhost:8000

Swagger docs: http://localhost:8000/docs

## Architecture

```
/oskin_backend/
    /app/
        /api/
            /routes/        # auth, fields, diseases, scans, suppliers, products, orders, weather, calculator, chat
        /core/              # config, security (JWT, bcrypt)
        /db/                # session, base, seed
        /models/            # SQLAlchemy ORM models
        /schemas/           # Pydantic v2 schemas
        /services/          # business logic
        main.py
    /alembic/               # migrations
    alembic.ini
    requirements.txt
    Dockerfile
    docker-compose.yml
```

## API Endpoints

### Auth
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /auth/register | Register new user |
| POST | /auth/login | Login → returns access + refresh tokens |
| POST | /auth/refresh | Refresh access token |
| GET | /auth/me | Current user profile |

### Fields
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /fields | Create field |
| GET | /fields | List user fields |
| GET | /fields/{id} | Get field |
| DELETE | /fields/{id} | Delete field |

### Diseases
| GET | /diseases | List all diseases |
| GET | /diseases/{id} | Get disease detail |

### Scans
| POST | /scans | Submit scan result |
| GET | /scans | List user scans |
| GET | /scans/{id} | Get scan |

### Suppliers & Products
| GET | /suppliers | List suppliers |
| GET | /products | List products |
| GET | /products/{id} | Get product |

### Orders
| POST | /orders | Create order from cart |
| GET | /orders | List user orders |
| POST | /orders/{id}/pay | Pay via mock Kaspi QR |

### Weather & Risk
| GET | /weather/{field_id} | Weather logs for field |
| GET | /risk/{field_id} | Disease risk score for field |

### Calculator
| POST | /calculator/roi | Calculate ROI for treatment |

### AI Agronomist
| POST | /chat | Context-aware agronomy advice |

## Business Logic

**Risk Engine**: Evaluates weather data per field:
- humidity > 80% AND temperature 18–24°C → +70 risk points
- precipitation > 10mm → +10 risk points
- Risk levels: LOW (<40), MEDIUM (40–69), HIGH (≥70)

**Economic Calculator**:
- `net_benefit = (revenue × loss_percent / 100) - treatment_cost`
- `roi_percentage = (net_benefit / treatment_cost) × 100`

**AI Agronomist** (rule-based):
1. If recent scan with disease → treatment advice for that disease
2. If field risk > 60 → weather-based risk warning
3. Otherwise → generic agronomy best practices

**Payment**: Mock Kaspi QR — sets order status to `paid`.

## Security

- JWT access tokens: 15 min expiry
- JWT refresh tokens: 7 days expiry
- Passwords hashed with bcrypt
- All endpoints except `/auth/register` and `/auth/login` require Bearer token
- CORS enabled for all origins

## Seed Data

Automatically applied on startup:
- 5 wheat diseases (Septoria, Rust, Fusarium, Powdery Mildew, Root Rot)
- 5 suppliers across Kazakhstan cities
- 10 fungicide products
