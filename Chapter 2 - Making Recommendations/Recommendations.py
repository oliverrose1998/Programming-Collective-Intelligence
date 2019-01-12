import numpy as np

critics = { 'Alfie' :{'Pulp Fiction':5,'Up':2, 'The Devil Wears Prada':3, 'Chalet Girl':3,'Where the Wild Things are':2},
           'Lottie' :{'Pulp Fiction':2,'Up':3, 'The Devil Wears Prada':4, 'Chalet Girl':5,'Where the Wild Things are':3},
           'Ben' :{'Pulp Fiction':3,'Up':4, 'The Devil Wears Prada':3, 'Chalet Girl':2,'Where the Wild Things are':4},
           'Freddie' :{'Pulp Fiction':4,'Up':5, 'The Devil Wears Prada':2, 'Chalet Girl':3,'Where the Wild Things are':5},
           'Olly' :{'Pulp Fiction':4,'Up':1, 'The Devil Wears Prada':2, 'Chalet Girl':3,'Where the Wild Things are':2},
           'Hannah' :{'Pulp Fiction':1,'Up':5, 'The Devil Wears Prada':4, 'Chalet Girl':5,'Where the Wild Things are':3}}

def Euclidean_Distance(prefs, person1, person2):
    #Get list of shared items
    si = {}
    for item in prefs[person1]:
        if item in prefs[person2]:
            si[item] = 1
            
    #If they have no items in common return 0
    if len(si) == 0: return 0
    
    #Add squares of all differences
    sum_of_squares = sum([pow(prefs[person1][item]-prefs[person2][item],2) for item in si])
    
    return 1/(1+np.sqrt(sum_of_squares))

def Pearson_Correlation(prefs, person1, person2):
    #List of mutually rated items
    si = {}
    for item in prefs[person1]:
        if item in prefs[person2]:si[item] = 1
            
    #Find number of elements
    n = len(si)
    
    #If no ratng in common, return 0
    if n == 0: return 0
    
    #Add all preferences
    sum1 = sum([prefs[person1][it] for it in si])
    sum2 = sum([prefs[person2][it] for it in si])
    
    #Sum squares
    sum1Sq = sum([pow(prefs[person1][it],2) for it in si])
    sum2Sq = sum([pow(prefs[person2][it],2) for it in si])
    
    #Sum products
    pSum = sum([prefs[person1][it]*prefs[person2][it] for it in si])
    
    #Calculate Pearson score
    num = pSum - (sum1*sum2/n)
    den = np.sqrt((sum1Sq - pow(sum1,2)/n)*(sum2Sq-pow(sum2,2)/n))
    if den ==0:return 0
    
    a = num/den
    return a

def Best_critic (prefs,person):
    scores = [(Pearson_Correlation(prefs,person, other), other) for other in prefs if other != person]
    scores.sort()
    scores.reverse()
    return scores

def topMatches(prefs,person,n=5,similarity=Pearson_Correlation):
    '''
    Returns the best matches for person from the prefs dictionary. 
    Number of results and similarity function are optional params.
    '''

    scores = [(similarity(prefs, person, other), other) for other in prefs
              if other != person]
    scores.sort()
    scores.reverse()
    return scores[0:n]


def getRecommendations(prefs, person, similarity=Pearson_Correlation):
    '''
    Gets recommendations for a person by using a weighted average
    of every other user's rankings
    '''

    totals = {}
    simSums = {}
    for other in prefs:
    # Don't compare me to myself
        if other == person:
            continue
        sim = similarity(prefs, person, other)
        # Ignore scores of zero or lower
        if sim <= 0:
            continue
        for item in prefs[other]:
            # Only score movies I haven't seen yet
            if item not in prefs[person] or prefs[person][item] == 0:
                # Similarity * Score
                totals.setdefault(item, 0)
                # The final score is calculated by multiplying each item by the
                #   similarity and adding these products together
                totals[item] += prefs[other][item] * sim
                # Sum of similarities
                simSums.setdefault(item, 0)
                simSums[item] += sim
    # Create the normalized list
    rankings = [(total / simSums[item], item) for (item, total) in
                totals.items()]
    # Return the sorted list
    rankings.sort()
    rankings.reverse()
    return rankings


def transformPrefs(prefs):
    '''
    Transform the recommendations into a mapping where persons are described
    with interest scores for a given title e.g. {title: person} instead of
    {person: title}.
    '''

    result = {}
    for person in prefs:
        for item in prefs[person]:
            result.setdefault(item, {})
            # Flip item and person
            result[item][person] = prefs[person][item]
    return result


def calculateSimilarItems(prefs, n=10):
    '''
    Create a dictionary of items showing which other items they are
    most similar to.
    '''

    result = {}
    # Invert the preference matrix to be item-centric
    itemPrefs = transformPrefs(prefs)
    c = 0
    for item in itemPrefs:
        # Status updates for large datasets
        c += 1
       # if c % 100 == 0:
        #    print '%d / %d' % (c, len(itemPrefs))
        # Find the most similar items to this one
        scores = topMatches(itemPrefs, item, n=n, similarity=Euclidean_Distance)
        result[item] = scores
    return result


def getRecommendedItems(prefs, itemMatch, user):
    userRatings = prefs[user]
    scores = {}
    totalSim = {}
    # Loop over items rated by this user
    for (item, rating) in userRatings.items():
        # Loop over items similar to this one
        for (similarity, item2) in itemMatch[item]:
            # Ignore if this user has already rated this item
            if item2 in userRatings:
                continue
            # Weighted sum of rating times similarity
            scores.setdefault(item2, 0)
            scores[item2] += similarity * rating
            # Sum of all the similarities
            totalSim.setdefault(item2, 0)
            totalSim[item2] += similarity
    # Divide each total score by total weighting to get an average
    rankings = [(score / totalSim[item], item) for (item, score) in
                scores.items()]
    # Return the rankings from highest to lowest
    rankings.sort()
    rankings.reverse()
    return rankings

def loadMovieLens(path='./ml-100k'):
    #Get movie titles
    movies = {}
    for line in open(path+'/u.item'):
        (id,title) = line.split('|')[0:2]
        movies[id] = title
    
    #Load data
    prefs={}
    for line in open(path+'/u.data'):
        (user,movieid,rating,ts) = line.split('\t')
        prefs.setdefault(user,{})
        prefs[user][movies[movieid]] = float(rating)
    return prefs

                 
   
        