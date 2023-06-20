# Takes in a 1-dimensional array and a window size, and applies a moving
# average filter to data.
def moving_filter(data, window_size):
    i = 0
    moving_averages = []
    while i < len(data) - window_size + 1:
        this_window = data[i : i + window_size]

        window_average = sum(this_window) / window_size
        moving_averages.append(window_average)
        i += 1

    for i in range(window_size - 1):
        ind = len(data) - (window_size - i)
        moving_averages.append(np.mean(data[ind-window_size:ind]))

    return moving_averages

# Returns the ratio between peak amplitude and noise of the extra-cellular signal
def AmplitudeToNoiseRatio(recordings):
    amp = np.max(recordings, axis=1)
    dev = np.std(recordings, axis=1)
    
    return np.where(dev==0, 0, amp/dev)

# Returns position of local minima in 2nd half of an extra-cellular signal
# This position is correlated with the steeper decay region of the  corresponding intra-cellular signal
def LocalMinimaEAP(recordings):
    result = []
    
    for rec in recordings:
        result.append(np.argmax(rec)+200+np.argmin(rec[np.argmax(rec)+200:]))
        
    return np.array(result)

# Returns a measure of the asymmetry of the extra-cellular signal's probability distribution
def Skewness(recordings):
    return scipy.stats.skew(recordings, axis=1)

# Returns a measure of the peakedness or flatness of the extra-cellular signal's probability distribution.
def Kurtosis(recordings):
    return scipy.stats.kurtosis(recordings, axis=1)

# Returns the total energy contained in the extra-cellular signal
def Energy(recordings):
    return np.sum(recordings**2, axis=1)

# Returns positive peak of extra-cellular signal
def PositiveAmplitude(recordings):
    return np.max(recordings, axis=1)

# Returns negative peak of extra-cellular signal
def NegativeAmplitude(recordings):
    return np.min(recordings, axis=1)

# Returns ratio of negative and positive amplitude in extra-cellular signal
def MinMaxRatio(recordings):
    ratios = []
    
    for rec in recordings:
        ratios.append(np.min(rec)/np.max(rec))
        
    return np.array(ratios)
  
# Returns decay rate of an intra-cellular signal. Refers to slope of middle region of intra-cellular AP
def DecayRate(recordings):
    rates = []
    
    for rec in recordings:
        max_index = np.argmax(rec)
        dv = moving_filter(np.gradient(moving_filter(rec, 20)), 20)
        min_index = np.argmin(dv)
        half_index = int((max_index + min_index)/2)
        rates.append(dv[half_index])
        
    return np.array(rates)

# Returns slope of an intra-cellular signal after initial decay. Refers to slope of the end region of intra-cellular AP
def SlopeAfterDecay(recordings):
    slopes = []
    
    for rec in recordings:
        dv = moving_filter(np.gradient(moving_filter(rec, 20)), 20)
        slopes.append(np.min(dv))
            
    return np.array(slopes)

# Returns area under the curve for intra-cellular recordings
def AUC(recordings):
    auc = []
    
    for rec in recordings:
        auc.append(sklearn.metrics.auc([i for i in range(len(rec))], rec))
        
    return np.array(auc)

# Returns correlations between recordings and their delayed versions. Can be used for intra-cellular recordings
def AutoCorrelation(recordings):
    result = []
    
    for rec in recordings:
        result.append(np.correlate(rec, rec, mode='full'))
        
    return np.array(result)

# Returns slope before first peak of an intra-cellular signal
def SlopeBeforePeak(recordings):
    slopes = []
    
    for rec in recordings:
        dv = moving_filter(np.gradient(moving_filter(rec, 20)), 20)
        slopes.append(np.max(dv))
            
    return np.array(slopes)

# Returns 4 points (timestep locations) for every intra-cellular recording
# These 4 points roughly divide the intra-cellular signal into regions that are approximately linear
def ChangePoints(recordings):
    points = []
    
    for rec in recordings:
        pt = []
        grad = moving_filter(np.gradient(moving_filter(rec, 30)), 30)
        x2 = np.argmax(grad)
        x1 = int(x2/2)
        x = [i for i in range(x1, x2) if grad[i] > (grad[x1]+grad[x2])/2][0]
        pt.append(x)
        pt.append(np.argmax(rec))
        
        x2 = np.argmin(grad)
        x1 = int((np.argmax(rec) + x2)/2)
        x = [i for i in range(x1, x2) if grad[i] < (grad[x1]+grad[x2])/2][0]
        pt.append(x)
        
        x1 = np.argmin(grad)
        x2 = int((x1+len(rec)-1)/2)
        x = [i for i in range(x1, x2) if grad[i] > (grad[x1]+grad[x2])/2][0]
        pt.append(x)
        
        points.append(pt)
        
    return np.array(points)
