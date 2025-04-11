import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

aspirants = pd.read_csv("aspirants.csv")
mentors = pd.read_csv("mentors.csv")

encoder = LabelEncoder()

for col in ['preferred_subject', 'location', 'experience_level']:
    aspirants[f'{col}_encoded'] = encoder.fit_transform(aspirants[col])
for col in ['expertise_subject', 'location', 'suitable_level']:
    mentors[f'{col}_encoded'] = encoder.fit_transform(mentors[col])

mentors_renamed = mentors.copy()
mentors_renamed['preferred_subject'] = mentors['expertise_subject']
mentors_renamed['experience_level'] = mentors['suitable_level'] 
mentors_renamed['study_hours'] = mentors['available_hours']

def recommend_mentors(aspirants, mentors, top_n=3):
    recommendations = {}
    for idx, aspirant in aspirants.iterrows():
        aspirant_vec = aspirant[['preferred_subject_encoded', 'location_encoded', 'study_hours', 'experience_level_encoded']].values.reshape(1, -1)
        mentor_vecs = mentors_renamed[['preferred_subject_encoded', 'location_encoded', 'study_hours', 'experience_level_encoded']].values
        
        similarities = cosine_similarity(aspirant_vec, mentor_vecs)[0]
        top_indices = similarities.argsort()[-top_n:][::-1]
        
        recommendations[aspirant['aspirant_id']] = mentors.iloc[top_indices]['mentor_id'].tolist()
    return recommendations

mentor_recommendations = recommend_mentors(aspirants, mentors_renamed)

print("Mentor Recommendations:")
for idx, mentors_list in enumerate(mentor_recommendations.values(), start=1):
    aspirant = aspirants.iloc[idx - 1]
    print(f"Aspirant {idx} ({aspirant['preferred_subject']}, {aspirant['location']}) -> Recommended Mentors:")
    for i, mentor_id in enumerate(mentors_list, start=1):
        mentor = mentors[mentors['mentor_id'] == mentor_id].iloc[0]
        print(f"  {i}. Mentor {mentor_id} ({mentor['expertise_subject']}, {mentor['location']}) - Available {mentor['available_hours']} hrs/week")
