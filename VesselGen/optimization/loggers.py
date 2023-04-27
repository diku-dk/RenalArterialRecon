import logging
import sys
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(processName)s %(filename)s:%(lineno)s -- %(message)s",
    
    
    
    
    filemode="w",
    filename='logs.log',
)

logging.info('Useful message')
logging.debug('debug message')

