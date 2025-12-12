from config.settings import DEFAULT_PAGE_SIZE
PAGE_SIZE = DEFAULT_PAGE_SIZE

BASE_QUERY = """
SELECT ?person ?personLabel ?personDescription (YEAR(?person_dob) AS ?birthYear) 
       ?birthPlace ?birthPlaceLabel 
       ?country ?countryLabel 
       ?object ?objectLabel ?objectDescription 
       ?objectTypeLabel 
       ?relationshipLabel
WHERE {
    ?person wdt:P31 wd:Q5;
          wdt:P569 ?person_dob. 
     
    ##YEAR_FILTER_HOOK##  
    ##FIND_HOOK## 
    
    OPTIONAL { ?person wdt:P19 ?birthPlace. }
    OPTIONAL { ?person wdt:P27 ?country. }
    OPTIONAL { ?object wdt:P31 ?objectType. }
    
    SERVICE wikibase:label { 
        bd:serviceParam wikibase:language "en,vi". 
    }
}
ORDER BY ?person
"""

ADVANCED_QUERY = """
SELECT ?person ?personLabel ?personDescription (YEAR(?person_dob) AS ?birthYear) ?birthPlace ?birthPlaceLabel
?country ?countryLabel ?object ?objectLabel ?objectDescription ?objectType
WHERE {

  ?person wdt:P31 wd:Q5;     
          wdt:P27 wd:Q881.               # Quốc tịch Việt Nam
          
  ?person wdt:P569 ?person_dob.
  
  ##YEAR_FILTER_HOOK##

  VALUES ?relation {
    wdt:P40    # Con cái (Child) -> Để suy ra cháu chắt thì dùng Graph BFS
    wdt:P26    # Vợ chồng (Spouse)
    wdt:P3373  # Anh chị em (Sibling)
    wdt:P22    # Cha (Father)
    wdt:P25    # Mẹ (Mother)
    
    wdt:P108   # Chủ lao động/Công ty (Employer) -> Để suy ra đồng nghiệp
    wdt:P463   # Thành viên của (Member of) -> Đảng, Hội nhóm
    wdt:P1344  # Tham gia sự kiện/Dự án (Participant in)
    wdt:P1066  # Học trò của (Student of)
    wdt:P184   # Người hướng dẫn (Doctoral advisor)
    wdt:P69    # Học tại trường (Educated at) -> Để suy ra bạn học
  }
  
  ?person ?relation ?object.
  OPTIONAL { 
      ?person wdt:P19 ?birthPlace. 
      ?birthPlace rdfs:label ?birthPlaceLabel. 
    }
    OPTIONAL { 
      ?person wdt:P27 ?country. 
      ?country rdfs:label ?countryLabel. 
    }
    
  OPTIONAL { ?object wdt:P31 ?objectType. }

  SERVICE wikibase:label { bd:serviceParam wikibase:language "vi,en". }
}
"""


SPOUSE = """
{ ?person wdt:P26 ?object. BIND("spouse" AS ?relationshipLabel) }
"""

FATHER = """
{ ?person wdt:P22 ?object. BIND("father" AS ?relationshipLabel) }
"""

MOTHER = """
{ ?person wdt:P25 ?object. BIND("mother" AS ?relationshipLabel) }
"""

SIBLING = """
{ ?person wdt:P3373 ?object. BIND("sibling" AS ?relationshipLabel) }
"""

# --- Cầu nối (Hubs) ---
EDUCATION = """
{ ?person wdt:P69 ?object. BIND("educated_at" AS ?relationshipLabel) }
"""

EVENT_PARTICIPANT = """
{ 
  ?person wdt:P1344 ?object. 
  BIND("participant_in" AS ?relationshipLabel) 
}
"""
POLITICAL_PARTY = """
{ 
  ?person wdt:P102 ?object. 
  BIND("member_of" AS ?relationshipLabel) 
}
"""
RELIGION = """
{ 
  ?person wdt:P140 ?object. 
  BIND("religion" AS ?relationshipLabel)
}
"""

POLITICAL_IDEOLOGY = """
  ?person wdt:P1142 ?object. 
  BIND("political_ideology" AS ?relationshipLabel)
"""

MEMBER_OF_GROUP = """
{ 
  ?person wdt:P463 ?object. 
  BIND("member_of_group" AS ?relationshipLabel)
  BIND("group" AS ?relationshipLabel) 
}
"""
EMPLOYER = """
{ 
  ?person wdt:P108 ?object. 
  BIND("employer" AS ?relationshipLabel) 
}
"""
PERFORMER = """
{ 
  ?object wdt:P31 wd:Q11424. 
  ?object wdt:P161 ?person. 
  BIND("performed_by" AS ?relationshipLabel) 
}
"""
FILM_DIRECTOR = """
{ 
  ?object wdt:P57 ?person. 
  BIND("director" AS ?relationshipLabel)
}
"""
FILM_ACTOR = """
{ 
  ?object wdt:P31 wd:Q11424. 
  ?object wdt:P161 ?person. 
  BIND("acted_in" AS ?relationshipLabel) 
}
"""

FILM_SCREENWRITER = """
{ 
  ?object wdt:P58 ?person. 
  BIND("screenwriter" AS ?relationshipLabel)
}
"""
MUSIC_COMPOSER = """
{ 
  ?object wdt:P86 ?person. 
  BIND("composer" AS ?relationshipLabel)
}
"""
MUSIC_LYRICIST = """
{ 
  ?object wdt:P676 ?person. 
  BIND("lyricist" AS ?relationshipLabel)
}
"""
AUTHOR = """
{ 
  ?object wdt:P50 ?person. 
  BIND("author" AS ?relationshipLabel) 
}
"""
AWARD = """
{ 
  ?person wdt:P166 ?object. 
  BIND("award_received" AS ?relationshipLabel) 
}
"""

POSITION_HELD = """
{ 
  ?person wdt:P39 ?object. 
  BIND("position_held" AS ?relationshipLabel) 
}
"""
PARTNER = """
{ 
  ?person wdt:P451 ?object. 
  BIND("partner" AS ?relationshipLabel) 
}
"""

STUDENT_OF = """
{ 
  ?person wdt:P1066 ?object. 
  BIND("mentor_student" AS ?relationshipLabel)
}
"""
ADVISOR_OF = """
{ 
  ?object wdt:P184 ?person. 
  BIND("mentor_student" AS ?relationshipLabel)
}
"""
INFLUENCED_BY = """
{ 
  ?object wdt:P737 ?person. 
  BIND("influenced_by" AS ?relationshipLabel)
}
"""
# Truy vấn dùng làm thuộc tính, không làm node
INTEREST_FIELD = """
{ 
  ?person wdt:P101 ?object. 
  BIND("field_of_work" AS ?relationshipLabel) 
}
"""
INTEREST_SPORT = """
{ 
  ?person wdt:P641 ?object. 
  BIND("sport" AS ?relationshipLabel) 
}
"""

INTEREST_INSTRUMENT = """
{ 
  ?person wdt:P1303 ?object. 
  BIND("instrument" AS ?relationshipLabel) 
}
"""

INTEREST_GENRE = """
{ 
  ?person wdt:P136 ?object. 
  BIND("genre" AS ?relationshipLabel) 
}
"""

ALL_QUERIES =   {
        #"spouse": (SPOUSE, PAGE_SIZE),
        #"father": (FATHER, PAGE_SIZE),
        #"mother": (MOTHER, PAGE_SIZE),
        #"sibling": (SIBLING, PAGE_SIZE),
        "education": (EDUCATION, PAGE_SIZE),
        # "interest_field": (INTEREST_FIELD, PAGE_SIZE),
        "interest_sport": (INTEREST_SPORT, PAGE_SIZE),
        # "interest_instrument": (INTEREST_INSTRUMENT, PAGE_SIZE),
        # "interest_genre": (INTEREST_GENRE, PAGE_SIZE),
        # "event": (EVENT_PARTICIPANT, PAGE_SIZE),
        # "party": (POLITICAL_PARTY, PAGE_SIZE),
        # "group": (MEMBER_OF_GROUP, PAGE_SIZE),
        "employer": (EMPLOYER, PAGE_SIZE),
        # "performer": (PERFORMER, PAGE_SIZE),
        # "film_actor": (FILM_ACTOR, PAGE_SIZE),
        # "film_director": (FILM_DIRECTOR, PAGE_SIZE),
        # "film_screenwriter": (FILM_SCREENWRITER, PAGE_SIZE),
        # "music_composer": (MUSIC_COMPOSER, PAGE_SIZE),
        # "music_lyricist": (MUSIC_LYRICIST, PAGE_SIZE),
        # "author": (AUTHOR, PAGE_SIZE),
        "award": (AWARD, PAGE_SIZE),
        # "position": (POSITION_HELD, PAGE_SIZE),
        # "partner": (PARTNER, PAGE_SIZE),
        # "student": (STUDENT_OF, PAGE_SIZE),
        # "advisor": (ADVISOR_OF, PAGE_SIZE),
        # "influenced": (INFLUENCED_BY, PAGE_SIZE),
        # "religion": (RELIGION, PAGE_SIZE),
        # "ideology": (POLITICAL_IDEOLOGY, PAGE_SIZE),
}
