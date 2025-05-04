import re

# Extraer los IDs de las URLs
def extract_id(url):
    try:
      return re.search(r'(\d+)/?$', url).group(1)  # Extrae el número al final
    except :
      return "0"

