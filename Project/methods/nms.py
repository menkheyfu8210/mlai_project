def IOU(p1, p2):
    # x-y coordinates of the detected rectangles
    x1_tl = p1[0]
    x2_tl = p2[0]
    x1_br = p1[0] + p1[3]
    x2_br = p2[0] + p2[3]
    y1_tl = p1[1]
    y2_tl = p2[1]
    y1_br = p1[1] + p1[4]
    y2_br = p2[1] + p2[4]
    # Overlap area
    xOverlap = max(0, min(x1_br, x2_br)-max(x1_tl, x2_tl))
    yOverlap = max(0, min(y1_br, y2_br)-max(y1_tl, y2_tl))
    overlapArea = xOverlap * yOverlap
    area1 = p1[3] * p1[4]
    area2 = p2[3] * p2[4]
    totalArea = area1 + area2 - overlapArea
    return overlapArea / float(totalArea)

#   Non-maximum suppression
#
# 1. Select from B the proposal with the highest confidence score, remove it 
#    from B, and add it to an initially empty list D.
# 2. Compare the proposal with all other proposals in B by computing the ratio
#    between the overlapping area and the total area (IOU).
# 3. Remove from B all the proposals with an IOU greater than some threshold.
# 4. Repeat until B is empty.

def nms(B, threshold=.5):
    if len(B) == 0:
        return []
    # Sort the proposals based on confidence score
    B = sorted(B, key=lambda B: B[2], reverse=True)

    D=[]
    D.append(B[0])
    del B[0]
    # Keep only the proposals whose overlapping area with other proposals is 
    # lower than the threshold
    for i, b in enumerate(B):
        for d in D:
            if IOU(b, d) > threshold:
                del B[i]
                break
        else:
            D.append(b)
            del B[i]
    return D