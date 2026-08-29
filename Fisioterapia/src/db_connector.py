"""
Conector opcional para Supabase.

Para usarlo:
1. Copia .env.example a .env
2. Completa SUPABASE_URL y SUPABASE_KEY
3. Instala supabase y python-dotenv.
"""

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def get_supabase_client():
    """Devuelve un cliente Supabase si las credenciales están configuradas."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        return None

    try:
        from supabase import create_client
        return create_client(url, key)
    except ImportError:
        return None
