function measure = getMeasure(response,row,col)
 
      [rs,cs]=ndgrid((1-row):(size(response,1)-row),(1-col):(size(response,2)-col));
      sum_response = sum(response(:));
      measure = sum(sum(abs(rs).*abs(cs).*response))/sum_response;
     
end

