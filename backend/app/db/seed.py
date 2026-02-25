from app.db.session import SessionLocal
from app.models.disease import Disease
from app.models.supplier import Supplier
from app.models.product import Product


def seed():
    db = SessionLocal()
    try:
        if db.query(Disease).count() > 0:
            print("Seed already applied.")
            return

        diseases = [
            Disease(
                name="Septoria Leaf Blotch",
                description="Fungal disease causing brown lesions on wheat leaves.",
                symptoms="Small brown spots with yellow halos on lower leaves, spreading upward.",
                causes="Zymoseptoria tritici fungus, favored by wet cool weather.",
                treatment_plan="Apply fungicide (tebuconazole or propiconazole) at first sign of infection.",
                prevention="Use resistant varieties, rotate crops, remove infected debris.",
            ),
            Disease(
                name="Wheat Rust (Yellow Rust)",
                description="Airborne fungal disease forming yellow-orange pustules on wheat.",
                symptoms="Yellow-orange striped pustules on leaves, stunted growth.",
                causes="Puccinia striiformis, spreads via wind spores in cool humid weather.",
                treatment_plan="Apply triazole or strobilurin fungicides immediately.",
                prevention="Plant resistant varieties, early sowing, monitor regularly.",
            ),
            Disease(
                name="Fusarium Head Blight",
                description="Fungal disease affecting wheat heads, reduces yield and grain quality.",
                symptoms="Bleached spikelets, pink or orange mold on grain.",
                causes="Fusarium graminearum, promoted by warm humid conditions during flowering.",
                treatment_plan="Apply fungicide (metconazole) at flowering. Remove infected plants.",
                prevention="Crop rotation with non-host crops, resistant varieties.",
            ),
            Disease(
                name="Powdery Mildew",
                description="Fungal disease forming white powdery coating on wheat surfaces.",
                symptoms="White powdery patches on leaves and stems.",
                causes="Blumeria graminis, favored by moderate temperatures and high humidity.",
                treatment_plan="Fungicides containing triadimenol or fenpropimorph.",
                prevention="Avoid dense planting, ensure airflow, use resistant cultivars.",
            ),
            Disease(
                name="Root Rot",
                description="Soilborne disease attacking wheat root system.",
                symptoms="Browning of roots, wilting, poor tillering, yellowing leaves.",
                causes="Bipolaris sorokiniana or Pythium spp., promoted by waterlogged soils.",
                treatment_plan="Seed treatment with fungicide, improve drainage, biological control.",
                prevention="Good soil drainage, balanced fertilization, crop rotation.",
            ),
        ]
        db.add_all(diseases)

        suppliers = [
            Supplier(
                name="AgroKaz Supply",
                city="Almaty",
                contact_phone="+77012345678",
                whatsapp_link="https://wa.me/77012345678",
                external_url="https://agrokaz.kz",
            ),
            Supplier(
                name="NurAgro",
                city="Nur-Sultan",
                contact_phone="+77172345678",
                whatsapp_link="https://wa.me/77172345678",
                external_url="https://nuragro.kz",
            ),
            Supplier(
                name="StepAgroTech",
                city="Kostanay",
                contact_phone="+77142345678",
                whatsapp_link="https://wa.me/77142345678",
                external_url="https://stepagrotech.kz",
            ),
            Supplier(
                name="Zhuldyz Agro",
                city="Shymkent",
                contact_phone="+77252345678",
                whatsapp_link="https://wa.me/77252345678",
                external_url="https://zhuldyzagro.kz",
            ),
            Supplier(
                name="DalaFarm",
                city="Pavlodar",
                contact_phone="+77182345678",
                whatsapp_link="https://wa.me/77182345678",
                external_url="https://dalafarm.kz",
            ),
        ]
        db.add_all(suppliers)
        db.flush()

        products = [
            Product(
                supplier_id=suppliers[0].id,
                name="Tebucon 250 EW",
                active_ingredient="Tebuconazole 250 g/L",
                price=4500.0,
                volume="1L",
                description="Systemic fungicide against leaf diseases in wheat.",
            ),
            Product(
                supplier_id=suppliers[0].id,
                name="Propimax EC",
                active_ingredient="Propiconazole 250 g/L",
                price=3800.0,
                volume="1L",
                description="Effective against rust and septoria in cereals.",
            ),
            Product(
                supplier_id=suppliers[1].id,
                name="Amistik Plus",
                active_ingredient="Azoxystrobin + Cyproconazole",
                price=6200.0,
                volume="1L",
                description="Broad spectrum fungicide for wheat and barley.",
            ),
            Product(
                supplier_id=suppliers[1].id,
                name="Falcon 460 EC",
                active_ingredient="Spiroxamine + Tebuconazole + Triadimenol",
                price=5500.0,
                volume="1L",
                description="Triple action fungicide for cereal diseases.",
            ),
            Product(
                supplier_id=suppliers[2].id,
                name="Topsin M 70 WP",
                active_ingredient="Thiophanate-methyl 700 g/kg",
                price=2900.0,
                volume="500g",
                description="Protective and curative fungicide.",
            ),
            Product(
                supplier_id=suppliers[2].id,
                name="Granivit FS",
                active_ingredient="Thiram + Carbendazim",
                price=1800.0,
                volume="1L",
                description="Seed treatment fungicide against soilborne diseases.",
            ),
            Product(
                supplier_id=suppliers[3].id,
                name="Metco 90 WG",
                active_ingredient="Metconazole 90 g/kg",
                price=7200.0,
                volume="500g",
                description="Recommended for Fusarium head blight control.",
            ),
            Product(
                supplier_id=suppliers[3].id,
                name="Baytan Universal",
                active_ingredient="Triadimenol + Fuberidazole",
                price=3400.0,
                volume="1L",
                description="Seed treatment for powdery mildew and smut diseases.",
            ),
            Product(
                supplier_id=suppliers[4].id,
                name="Folicur BT",
                active_ingredient="Tebuconazole 250 g/L",
                price=4100.0,
                volume="1L",
                description="Systemic fungicide for cereal rust control.",
            ),
            Product(
                supplier_id=suppliers[4].id,
                name="Amistar Xtra",
                active_ingredient="Azoxystrobin + Cyproconazole",
                price=6800.0,
                volume="1L",
                description="Premium fungicide with curative and protective action.",
            ),
        ]
        db.add_all(products)
        db.commit()
        print("Seed completed successfully.")
    except Exception as e:
        db.rollback()
        print(f"Seed error: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
