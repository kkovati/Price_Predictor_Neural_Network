import sys


class LoadingBar:
    
    def __init__(self, size, message):
        self.size = size
        self.message = message
        self.bar_length = 30
        self.index = 0
        
    def __call__(self, end=False): 
        if end:
            sys.stdout.write('\r' +'100% '+ self.message)
            print('')
            return
        self.index += 1 
        if self.index % (self.size / 100) > 1:
            return
        i = self.index
        line = (str(int(i * 100 / self.size)) + '%' + ' ' + self.message +
                ' |' + '|' * int(self.bar_length * i / self.size) + 
                '.' * int(self.bar_length * (1 - (i / self.size))) + 
                '|   ') 
        sys.stdout.write('\r' + line)

  
