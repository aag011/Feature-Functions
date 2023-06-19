# Function to compute gradient of an array
def Gradient(arr):
    grad = []
    delta = 20
    for i in range(len(arr)):
        if(i+delta < len(arr)):
            grad.append(5000*(arr[i+delta]-arr[i])/delta)
        else:
            grad.append(5000*(arr[i]-arr[i-delta])/delta)
            
    return np.array(grad)

# Returns positive peak of extra-cellular signal
def PositiveAmplitude(recordings):
    return np.max(recordings, axis=1)

# Returns negative peak of extra-cellular signal
def NegativeAmplitude(recordings):
    return np.min(recordings, axis=1)

# Takes in a 1 dimensional numpy array and a window size, and applies a moving
# average filter to the data.
def moving_filter(data, window_size):
    i = 0
    moving_averages = []
    while i < len(data) - window_size + 1:
        this_window = data[i : i + window_size]

        window_average = sum(this_window) / window_size
        moving_averages.append(window_average)
        i += 1

    for i in range(window_size - 1):
        moving_averages.append(data[len(data) - (window_size - i)])

    return moving_averages

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
        dv = Gradient(moving_filter(rec, 20))
        min_index = np.argmin(dv)
        half_index = int((max_index + min_index)/2)
        rates.append(dv[half_index])
        
    return np.array(rates)

# Returns slope of an intra-cellular signal after initial decay. Refers to slope of the end region of intra-cellular AP
def SlopeAfterDecay(recordings):
    slopes = []
    
    for rec in recordings:
        dv = Gradient(moving_filter(rec, 20))
        slopes.append(np.min(dv))
            
    return np.array(slopes)
