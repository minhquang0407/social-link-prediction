
class QueryTemplates:
    """
    Dùng để lưu trữ các chuỗi truy vấn (Constants).
    """
    def __init__(self, page_size = 20000):
        self.page_size = page_size
    
    BASE = """
    SELECT 
      ?person ?personLabel ?personDescription(YEAR(?person_dob) AS ?birthYear) ?birthPlaceLabel
      ?countryLabel ?object ?objectLabel ?objectDescription ?relationshipLabel
    WHERE {
        # 1. Neo & Lọc cơ bản
        ?person wdt:P31 wd:Q5. 
        ?person wdt:P569 ?person_dob.
        
        ##YEAR_FILTER_HOOK##
        
        ##FIND_HOOK##  
                
        OPTIONAL { 
          ?person wdt:P19 ?birthPlace. 
          ?birthPlace rdfs:label ?birthPlaceLabel. 
          FILTER(LANG(?birthPlaceLabel) = "vi") # Ưu tiên tiếng Việt
        }
    
        OPTIONAL { 
          ?person wdt:P27 ?country. 
          ?country rdfs:label ?countryLabel. 
          FILTER(LANG(?countryLabel) = "vi") # Ưu tiên tiếng Việt
        }

        SERVICE wikibase:label { 
            bd:serviceParam wikibase:language "vi,en". 
        }
    }
    ORDER BY ?person
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
      BIND("member_of" AS ?relationshipLabel) 
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
      BIND("student_of" AS ?relationshipLabel)
    }
    """
    ADVISOR_OF = """
    { 
      ?object wdt:P184 ?person. 
      BIND("advisor_of" AS ?relationshipLabel)
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
    def get_all_queries(self):
        return {
            "spouse": (self.SPOUSE, self.page_size),
            "father": (self.FATHER, self.page_size),
            "mother": (self.MOTHER, self.page_size),
            "sibling": (self.SIBLING, self.page_size),
            "education": (self.EDUCATION, self.page_size),
            "interest_field": (self.INTEREST_FIELD, self.page_size),
            "interest_sport": (self.INTEREST_SPORT, self.page_size),
            "interest_instrument": (self.INTEREST_INSTRUMENT, self.page_size),
            "interest_genre": (self.INTEREST_GENRE, self.page_size),
            "event": (self.EVENT_PARTICIPANT, self.page_size),
            "party": (self.POLITICAL_PARTY, self.page_size),
            "group": (self.MEMBER_OF_GROUP, self.page_size),
            "employer": (self.EMPLOYER, self.page_size),
            "performer": (self.PERFORMER, self.page_size),
            "film_actor": (self.FILM_ACTOR, self.page_size),
            "film_director": (self.FILM_DIRECTOR, self.page_size),
            "film_screenwriter": (self.FILM_SCREENWRITER, self.page_size),
            "music_composer": (self.MUSIC_COMPOSER, self.page_size),
            "music_lyricist": (self.MUSIC_LYRICIST, self.page_size),
            "author": (self.AUTHOR, self.page_size),
            "award": (self.AWARD, self.page_size),
            "position": (self.POSITION_HELD, self.page_size),
            "partner": (self.PARTNER, self.page_size),
            "student": (self.STUDENT_OF, self.page_size),
            "advisor": (self.ADVISOR_OF, self.page_size),
            "influenced": (self.INFLUENCED_BY, self.page_size),
            "religion": (self.RELIGION, self.page_size),
            "ideology": (self.POLITICAL_IDEOLOGY, self.page_size),
            
        }

