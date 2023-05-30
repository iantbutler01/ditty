def convert_seconds_to_string_time(seconds): 
    day = seconds // (24 * 3600) 
    seconds = seconds % (24 * 3600) 
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
      
    return "%d days, %02d hours, %02d minutes, %02d seconds" % (day, hour, minutes, seconds) 
