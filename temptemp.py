from datetime import datetime
import astropy.time

# 示例日期
date = datetime(2015, 10, 1, 12, 0, 0)  # 2023年10月1日中午12:00

# 转换为儒略日（JD）
jd = astropy.time.Time(date, format='datetime').jd

# 转换为简化儒略日（MJD）
mjd = jd - 2400000.5

jd2 = jd - 1721424.5

print("儒略日（JD）：", jd)
print("简化儒略日（MJD）：", mjd)
print(jd2)
