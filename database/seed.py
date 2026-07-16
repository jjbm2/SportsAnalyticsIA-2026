from database.database import get_session
from database.models import Sport


def seed_sports():
    session = get_session()

    try:
        sports = ["Fútbol", "Béisbol", "Basketball", "NFL"]

        for sport_name in sports:
            existing = session.query(Sport).filter_by(name=sport_name).first()

            if not existing:
                session.add(Sport(name=sport_name))

        session.commit()

    except Exception:
        session.rollback()
        raise

    finally:
        session.close()