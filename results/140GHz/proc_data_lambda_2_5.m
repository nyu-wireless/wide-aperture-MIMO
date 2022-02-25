clc 
close all
clear all

for ndata=1:2
    switch ndata 
        case 1    
            data = table2array(readtable('err_results_no_foliage_no_diffraction.csv'));
        case 2    
            data = table2array(readtable('err_results_add_foliage_add_diffraction.csv'));
    end
    
    remove_anomaly = 0;
    
    if remove_anomaly
        data(find(data(:,3)>20),:) = zeros(1,size(data,2));
    end
     
    links = 43;
    
    str_col = ['ro';'bx';'gs'];
    str_col_cdf = ['r';'b';'g'];
    str_cdf_colors = ['r';'b';'g'];
    dis=[2, 5, 10, 50, 100];
    
    for k = 1:5 
        switch k 
            case 1    
                title_fig = '2\lambda';
                save_fig = '2_lambda.fig';
            case 2    
                title_fig = '5\lambda';
                save_fig = '5_lambda.fig';
            case 3    
                title_fig = '10\lambda';     
                save_fig = '10_lambda.fig';
            case 4    
                title_fig = '50\lambda';     
                save_fig = '50_lambda.fig';
            case 5    
                title_fig = '100\lambda';     
                save_fig = '100_lambda.fig';
        end
        %sgtitle(title_fig,'fontsize',40)
        figure(1)
        subplot(2,5,(ndata-1)*5+k)
        for i = 1:3
            I = find(data(:,2)==dis(k));
            [f,x] = ecdf(data(I,i+3));
            semilogx(x,f,str_col_cdf(i,:),'linewidth',3)
            hold on
        end
        if ndata==2 && k==3
            xlabel('Error','FontSize',5);
        end
        if k==1
            ylabel('eCDF','FontSize',5)
        end

        if ndata==1 && k==1
            legend('RM','PWA','Const','','Location','northwest','FontSize',12)
        end
        set(gca,'fontsize',16,'XLim',[1e-6 1e0]);
        frame_h = get(handle(gcf),'JavaFrame');
        set(frame_h,'Maximized',1);
        grid on 
        if ndata == 1
            title(title_fig, 'FontSize', 16)
        end 
    end
end